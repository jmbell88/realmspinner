"""The ModernGL viewport: PBR shading, camera, gizmos, capture.

Loading a GLB is not done here: it moved to ``realmspinner/kernels/geom3d/gltf.py``
on 2026-09-17, and this package draws the model that returns (the 2026-10-11
audit, finding create-26, found this line still claiming it).

Nothing here imports imgui or pygame. The viewer renders into framebuffers and
answers questions about rays and bones; :mod:`realmspinner.studio.viewer_embed` is
what turns that into a panel the UI can show and click on. Keeping the split
means the whole renderer is testable against a standalone GL context with no
window at all -- which is what the golden-image tests do.
"""
