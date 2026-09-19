"""A regular package, so ``tests.message_golden`` cannot be shadowed.

Kaggle's Python 3.12 image has a regular top-level ``tests`` package in
site-packages. While this directory was a namespace package, that one won
the import from anywhere on sys.path, and the on-card golden guard died with
"No module named 'tests.message_golden'" before any training (headline seed
2, first attempt). A regular package found earlier on sys.path takes
precedence, and every test file puts the repository root first.
"""
