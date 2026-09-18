"""Muse: ACE-Step music generation. Its output is a job row, not a document.

The loop finder and waveform reduction are the headless ``engine/``. What else
is Muse's own lives here: the controller (``mode``), what it remembers
between frames (``state``), a take's file layer (``fileio``) and the drawn
half under ``ui/``."""
