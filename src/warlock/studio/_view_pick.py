"""What the cursor is over: object picking, projection and element picking.

Split out of :mod:`~warlock.studio.clay_view` as pure code motion. :class:`Hit`
lives here rather than in the viewport module because these methods construct it
at runtime, and a mixin may not import ``clay_view`` outside ``TYPE_CHECKING``;
``clay_view`` re-exports the name, so every existing importer is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .viewer import picking

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .clay_view import ClayView


@dataclass(frozen=True)
class Hit:
    """What a viewport ray found: the object, the distance, and the face.

    The distance is what element picking's occlusion test compares against --
    "no further from the eye than the surface you can see there" -- so it has
    to come back with the hit rather than being re-derived from a second ray
    cast that could disagree with the first.
    """

    uid: int
    t: float
    face: int


class PickOps:
    """``ClayView``'s picking. See the module docstring."""

    # -- picking -----------------------------------------------------------

    def pick_face(self: ClayView, doc: Any, local: tuple[float, float]) -> Hit | None:
        """What a click lands on: which object, how far, and which *face*.

        Object space per object, with the AABB prefilter, so the ray goes
        through one inverse transform per object rather than every triangle
        going through the forward one.

        The triangulation comes from ``adjacency.cached_triangulation``, keyed
        weakly on the mesh -- not from the GPU cache entry, which is also keyed
        on the palette and would therefore be rebuilt by a material edit that
        cannot possibly have moved a triangle. That cache is finally what makes
        ``tri_face`` earn its place: it maps the triangle the ray hit back to
        the n-gon the user thinks they clicked.
        """
        origin, direction = self._ray(local)
        best: Hit | None = None
        for obj in doc.objects:
            if not obj.visible:
                continue
            hit = self._pick_face_on(obj, origin, direction)
            if hit is not None and (best is None or hit.t < best.t):
                best = hit
        return best

    def _pick_face_on(self: ClayView, obj: Any, origin: Any, direction: Any) -> Hit | None:
        """One object's own nearest face hit, ignoring every other object.

        Split out of :meth:`pick_face` for face-mode picking under X-ray (the
        2026-09-13 audit's clay-02): that ray loop already stops at whichever
        object's surface is nearest the eye, so a face behind it never had a
        ray cast against it at all, X-ray or not. Vertex and edge mode do not
        have this hole -- they already rank a *per-object* candidate by depth
        in :meth:`pick_element` and simply skip the depth filter under X-ray --
        so face mode needs the same per-object candidate this returns.
        """
        from ..kernels.mesh.adjacency import cached_positions_f8, cached_triangulation

        tris, tri_face = cached_triangulation(obj.mesh)
        positions = cached_positions_f8(obj.mesh)
        hit = picking.ray_object(
            origin,
            direction,
            self._world(obj),
            positions,
            tris,
            # Positions and tree both come from the same frozen mesh, so
            # they cannot disagree about the geometry -- which is the whole
            # precondition the narrowed sweep rests on.
            bvh=picking.cached_bvh(obj.mesh, positions, tris),
        )
        if hit is None:
            return None
        face = int(tri_face[hit[1]]) if len(tri_face) else -1
        return Hit(uid=obj.uid, t=float(hit[0]), face=face)

    def pick(self: ClayView, doc: Any, local: tuple[float, float]) -> int | None:
        """Which object a click lands on. -> its uid, or None.

        Kept as the object-mode entry point, and kept returning a bare uid:
        that is what selection in object mode is, and widening it would make
        every caller unpack a record to ignore two thirds of it.
        """
        hit = self.pick_face(doc, local)
        return None if hit is None else hit.uid

    def screen_of(self: ClayView, doc: Any, uid: int) -> Any:
        """One object's vertices projected into the viewport, for element picking.

        Built here rather than in ``clay/pick.py`` because it is the only step
        that needs the camera; everything the picking rules actually decide is
        pure numpy over what this returns.

        Cached per object on ``(id(mesh), transform, camera, rect)``, which is
        every input the projection has. Hover runs on every mouse move, and a
        200k-vertex import reprojected per move is the difference between a
        viewport and a slideshow -- while a camera that has not moved makes the
        key hit and the whole thing free.
        """
        from ..kernels.mesh import pick as bp

        obj = doc.by_uid(uid)
        width, height = int(max(self._rect[2], 1)), int(max(self._rect[3], 1))
        self.camera.aspect = width / max(height, 1)
        matrix = self._world(obj)
        key = (
            id(obj.mesh),
            matrix.tobytes(),
            (self.camera.theta, self.camera.phi, self.camera.distance),
            tuple(self.camera.target),
            (width, height),
            # The projection *kind* moves every projected point without moving
            # the camera, so leaving it out let a Ctrl+5 toggle serve stale
            # positions to pick, hover and the marquee until the camera moved.
            bool(self.camera.orthographic),
        )
        cached = self._screens.get(uid)
        if cached is not None and cached[0] == key:
            return cached[1]
        screen = bp.project(
            obj.mesh.positions,
            matrix,
            self.camera.projection() @ self.camera.view(),
            self.camera.position,
            width,
            height,
        )
        # The mesh rides along as the ``_view_cache._Entry`` pin does: its id is
        # in the key, and an id is only sound while the object it named is
        # alive -- a freed mesh's address coming back on new geometry would
        # otherwise match a stale projection.
        self._screens[uid] = (key, screen, obj.mesh)
        return screen

    def pick_element(
        self: ClayView, doc: Any, local: tuple[float, float], hit: Any = None
    ) -> tuple[int, int] | None:
        """What element is under the cursor: ``(uid, index)``, read through the mode.

        The *index* is a vertex index, an index into ``adjacency.edge_verts`` or
        a face index, depending on ``doc.element_mode`` -- one shape for all
        three, because every caller here does the same thing with it.

        The surface hit is passed in rather than recast so the occlusion test
        compares against the same ray the object pick used; recasting would let
        the two disagree by an ulp and make a vertex on the near face flicker
        in and out of pickability.
        """
        from ..kernels.mesh import pick as bp
        from ..kernels.mesh.adjacency import adjacency

        mode = doc.element_mode
        if mode == "object":
            return None
        xray = getattr(self, "xray", False)
        if hit is None:
            hit = self.pick_face(doc, local)
        # X-ray is see-through for the pick as well as the draw: with no
        # surface depth the nearest element wins wherever it sits, which is
        # what ``clay_state.xray`` and the Clay chapter say it does.
        depth = None if hit is None or xray else hit.t
        origin, direction = (None, None)
        if mode == "face" and xray:
            # Under X-ray, ``hit`` is only the *frontmost* object's ray hit --
            # ``pick_face`` stops there, so a face behind it was never even
            # tested. Recast per object below (the 2026-09-13 audit's clay-02)
            # instead of reusing ``hit``, the way vertex and edge already
            # rank a per-object candidate rather than trusting one shared hit.
            origin, direction = self._ray(local)

        best: tuple[float, int, int] | None = None
        for obj in doc.objects:
            if not obj.visible:
                continue
            screen = self.screen_of(doc, obj.uid)
            if mode == "vertex":
                index = bp.nearest_vertex(screen, local, surface_depth=depth)
            elif mode == "edge":
                index = bp.nearest_edge(
                    screen, adjacency(obj.mesh).edge_verts, local, surface_depth=depth
                )
            elif xray:
                face_hit = self._pick_face_on(obj, origin, direction)
                index = None if face_hit is None or face_hit.face < 0 else face_hit.face
            else:
                index = hit.face if hit is not None and hit.uid == obj.uid else None
                index = None if index is not None and index < 0 else index
            if index is None:
                continue
            if mode == "vertex":
                key = float(screen.depth[index])
            elif mode == "edge":
                # The mean of the endpoints' depths -- the same reading
                # ``nearest_edge`` compared against the surface. This was a
                # constant 0.0, so with two objects' edges under the cursor
                # the earlier one in ``doc.objects`` always won regardless of
                # which edge was nearer the camera.
                a, b = adjacency(obj.mesh).edge_verts[index]
                key = 0.5 * (float(screen.depth[a]) + float(screen.depth[b]))
            elif xray:
                # Multiple objects can each have a face candidate under
                # X-ray now (clay-02) -- rank by the ray's own hit distance,
                # the same "nearer wins" rule vertex/edge use their depth for.
                key = float(face_hit.t)
            else:
                # No X-ray: ``hit`` already names the one nearest object, so
                # only it ever reaches here with a face index.
                key = 0.0
            if best is None or key < best[0]:
                best = (key, obj.uid, int(index))
        return None if best is None else (best[1], best[2])

    def element_sel_for(self: ClayView, doc: Any, uid: int, index: int) -> Any:
        """One picked element as an :class:`~.clay.elements.ElementSel`."""
        from ..kernels.mesh import elements as el
        from ..kernels.mesh.adjacency import adjacency

        mode = doc.element_mode
        if mode == "vertex":
            return el.ElementSel(verts=[index])
        if mode == "edge":
            return el.ElementSel(edges=[adjacency(doc.by_uid(uid).mesh).edge_verts[index]])
        return el.ElementSel(faces=[index])
