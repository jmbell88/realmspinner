"""Clay: mesh primitives, booleans and element editing, one document mode.

The mesh engine itself is ``realmspinner.kernels.mesh`` -- shared, since the
character families build with it too. What is Clay's own lives here: the
document tabs and their state (``mode``, ``state``), the op registry
(``ops``), the viewport and panes (``ui/``) and the MCP tool surface
(``agent/``)."""
