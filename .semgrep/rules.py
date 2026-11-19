# ruleid: no-docstrings
"""Module docstring."""

import os
from pathlib import Path


def with_docstring():
    # ruleid: no-docstrings
    """Does a thing."""
    return 1


async def async_with_docstring():
    # ruleid: no-docstrings
    'single quoted'
    return 1


def without_docstring():
    # ok: no-docstrings
    x = "a string, not a docstring"
    return x


class WithDocstring:
    # ruleid: no-docstrings
    """A class."""


class WithBaseAndDocstring(Path):
    # ruleid: no-docstrings
    """A subclass."""


class WithoutDocstring:
    # ok: no-docstrings
    value = "not a docstring"


def local_imports():
    # ruleid: no-local-imports
    import json
    # ruleid: no-local-imports
    from collections import OrderedDict, defaultdict
    # ruleid: no-local-imports
    from . import sibling
    return json, OrderedDict, defaultdict, sibling


class Holder:
    def method(self):
        # ruleid: no-local-imports
        import re
        return re


# ok: no-local-imports
import sys

def path_checks():
    # ruleid: hardcoded-absolute-path
    open('/home/someone/songs/a.wav')
    # ruleid: hardcoded-absolute-path
    open("/mnt/c/Users/someone/a.wav")
    # ruleid: hardcoded-absolute-path
    open(r'C:\Users\someone\a.wav')
    # ok: hardcoded-absolute-path
    open('data/a.wav')
