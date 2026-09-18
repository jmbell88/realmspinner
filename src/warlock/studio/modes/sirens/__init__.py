"""Sirens: the chiptune tracker, one document mode.

The song document, synth and file format are the headless ``engine/``. What
else is Sirens's own lives here: the controller (``mode``) and the verbs it
delegates to (``edit``, ``play``, ``keys``), its state, file layer
(``fileio``), hint line, the one module that touches ``pygame.mixer``
(``audio``), and the drawn half under ``ui/``."""
