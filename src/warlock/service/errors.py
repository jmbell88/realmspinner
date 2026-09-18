"""What the service layer raises instead of ``HTTPException``.

One class per distinguishable outcome, not one per status code: the desktop UI
shows ``exc.message`` in a toast and mostly cares only that it failed.
``status`` is a fossil of the HTTP API these classes were first written for --
the routes and the ``_to_http`` mapping that read this attribute are gone
(``service/__init__.py`` names the two loopback clients that are the app's
whole outbound network today), and nothing left in the tree reads ``status``.
It stays on each class because the numbers still communicate the same rank
order to a reader (a 404-shaped refusal versus a 409-shaped one), and because
renumbering it now would be churn with no caller to fix.
"""

from __future__ import annotations


class ServiceError(Exception):
    """Base for every expected failure. ``message`` is user-facing."""

    status = 400

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        rows: tuple[str, ...] = (),
        packs: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.message = message
        # Which form control was at fault, when the caller can know. The UI
        # highlights it; HTTP has nowhere to put it and drops it.
        self.field = field
        # Which registry rows (``fetch.Entry.row_key``) would fix this, when the
        # refusal is "these weights are not on this host". Structure rather than
        # a second parse of ``message``: the pane offers an Install button that
        # has to pre-tick exact rows, and re-deriving them from the sentence is
        # the brittleness this field exists to avoid. Empty for every refusal
        # that is not about missing weights, which is most of them.
        self.rows = tuple(rows)
        # ``rows``' sibling for the *other* thing a job can be short of. A pack
        # (``packs.Pack.key``) is not a registry row -- there is nothing to
        # download, and nothing ``downloads.needed_gib`` can size -- so it gets
        # its own carrier rather than being folded into ``rows`` and losing the
        # distinction the pane needs to draw the right button (an "Install the
        # X pack" offer that routes to Settings -> Packs, not Settings ->
        # Models). Same shape, same reason: structure the pane reads rather
        # than a second parse of ``message``. Empty for every refusal that is
        # not "the code for this is not installed", which is most of them.
        self.packs = tuple(packs)


class NotFound(ServiceError):
    """No such job / pose / sheet / file. Also covers a malformed id: a caller
    that supplies one cannot tell the difference, and saying so leaks less."""

    status = 404


class Invalid(ServiceError):
    """The request is well-formed but its values are not usable."""

    status = 400


class Conflict(ServiceError):
    """The object exists but is in the wrong state (running, at its limit)."""

    status = 409


class NotReady(NotFound):
    """The artifact is not on disk yet, or is still being written.

    A subclass of NotFound because that is the status the routes returned for
    it, and because "not ready" and "not there" are the same fact to a caller
    that can only retry. It exists separately so the UI can say *why* a
    download is disabled instead of just greying it out.
    """


class TooLarge(ServiceError):
    """An upload is over its byte cap, refused before it is decoded."""

    status = 413


def invalid_from(exc: Exception, context: str, *, field: str | None = None) -> Invalid:
    """Wrap a library ``ValueError`` in a sentence that says what was refused.

    The service layer used to re-raise these as ``Invalid(str(exc))`` (E49),
    which put library text written for whoever wrote the library straight in
    front of the user: *joints payload requires a non-empty 'bones' list* is
    precise, mentions no control that exists on screen, and names a wire format
    the user has never seen. The detail is kept -- it is the only part that says
    *which* value -- but framed by a sentence naming the thing being done.

    ``field`` defaults to whatever the exception carries: :class:`~warlock.
    guidance.GuidanceError` names the control it came from, and passing the
    address through is the whole reason it does (S137). An explicit ``field``
    wins, for the callers that know better than the library does.
    """
    detail = str(exc).strip()
    carried = getattr(exc, "field", None)
    message = f"{context}: {detail}" if detail else context
    return Invalid(message, field=field or (carried if isinstance(carried, str) else None))


class Failed(ServiceError):
    """A subprocess or conversion that should have worked, didn't.

    Blender is installed and the bake still died, gltfpack ran and returned
    garbage. The user's only useful response is to retry or report it, which is
    exactly what a 500 said.
    """

    status = 500
