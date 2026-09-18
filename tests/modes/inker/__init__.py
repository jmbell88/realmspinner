"""Makes Inker's tests a package: other suites import its helpers as ``modes.inker.test_*``.

See ``tests/modes/__init__.py`` for why a bare basename import is not enough here.
``flourish/`` and ``walk/`` stay plain directories, each rooted on itself.
"""
