"""The Manual as text: loading chapters, parsing them, and the targets a
help button points at.

Splitting the Manual's text from its rendering is what lets a headless caller
(the CLI, ``doctor``, the tests) read a chapter without an imgui context in
the process -- ``loader`` and ``parser`` used to sit under ``studio/``, which
is a UI package answering a question that has nothing to do with drawing.
``manual/render.py`` stayed in ``studio/``, because drawing is the part that
really is UI.

``loader`` resolves the shipped chapters relative to its own file; this
package sits at the same depth under ``realmspinner/`` that it did before, which
is why the move needed no change to that walk.
"""
