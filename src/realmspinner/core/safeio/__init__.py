"""Writing to disk without leaving half a file behind, and reading formats
that lie about their size.

Every module here was under ``studio/`` until 2026-09-17, which is why the
non-UI half of this codebase grew its own copies: roughly fifteen hand-rolled
``os.replace`` stagings across ``db``, ``migrate``, ``judge``, the character
generators and four pipelines, each re-deriving the rule ``atomic`` already
states. They could not import it, because ``pipelines`` may not import
``studio`` -- a layering rule doing its job and producing duplication anyway,
because the shared thing was on the wrong side of the line.

``sizeguard``, ``zipguard``, ``xmlguard``, ``npyguard`` and ``pixelguard`` are
the reading half: a decompression bomb, a zip slip, an XML entity expansion
and a numpy pickle are all "this file is not what its header claims", and the
answer is a refusal with a field, not an exception from three layers down.
"""
