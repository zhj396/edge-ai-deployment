"""Single source of truth for the package version.

Three public subpackages (``src``, ``cli``, ``utils``) and the CLI banner in
``main.py`` all import from here, so the version is changed in exactly one
place. Bump here, then re-run the test suite — ``tests/test_utils.py`` asserts
that every public ``__version__`` agrees with this value.
"""
__version__ = "1.0.0"
