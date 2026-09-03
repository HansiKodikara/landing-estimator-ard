"""Put the ``src/`` directory on sys.path so scripts can ``import lze``.

Keeps the repo runnable without an editable install (``pip install -e .``),
which matters on a Raspberry Pi where you may just ``git clone`` and run.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
