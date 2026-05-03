"""Make the parent representation-learning folder importable from the tests."""

import os
import sys

PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
