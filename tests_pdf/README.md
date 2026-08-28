# PDF smoke tests

These tests require the optional `pdf` dependency group and are intentionally
kept outside the dependency-free `tests/` suite. GitHub Actions runs them on
Python 3.9 and 3.12 after installing `.[pdf]`.
