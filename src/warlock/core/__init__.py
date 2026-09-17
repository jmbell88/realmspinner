"""Layer 0: the foundation. No domain, no UI, no opinions about assets.

What belongs here is what every other layer may need and none of them may
own -- configuration, the job store, the memory and VRAM accounting, the
Windows job object every child process is born into, the atomic-write door.
The rule that decides membership is negative and easy to check: a module
here may import the standard library and nothing else of Warlock's, so
nothing in this package can drag a pipeline, a service or a window in
behind it.

See ``dev/RESTRUCTURE.md`` for the six layers and ``tests/test_layering.py``
for the pin that measures them.
"""
