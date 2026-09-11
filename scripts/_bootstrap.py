"""Put the ``src/`` directory on sys.path so scripts can ``import lze``.

Keeps the repo runnable without an editable install (``pip install -e .``),
which matters on a Raspberry Pi where you may just ``git clone`` and run.

Also holds :func:`load_model`, the one way every script loads the surrogate --
see its docstring for why that needs to be shared.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_model(path):
    """Load the surrogate, or exit saying how to get one.

    The trained model is deliberately git-ignored: it is a 2.5 MB regenerable
    binary, not source. So a fresh clone has no ``data/surrogate.joblib``, and
    loading it raised a bare ``FileNotFoundError`` traceback that named the
    missing path and nothing else. At a test site that reads as "the software
    is broken" rather than "run two commands first", which is an expensive
    misreading when there is hardware waiting.
    """
    from lze.model.surrogate import Surrogate

    p = Path(path)
    if not p.exists():
        sys.exit(
            f"No model at {path}\n\n"
            f"  The trained model is not in git -- it is a regenerable binary.\n"
            f"  Get one either way:\n\n"
            f"    copy data/surrogate.joblib from the release zip, or build it:\n"
            f"      python scripts/generate_dataset.py --n-flights 80 "
            f"--out data/dataset.npz\n"
            f"      python scripts/train_model.py --dataset data/dataset.npz "
            f"--out {path}\n"
        )
    try:
        return Surrogate.load(p)
    except Exception as exc:
        sys.exit(
            f"Could not load the model at {path}: "
            f"{type(exc).__name__}: {exc}\n\n"
            f"  If the file was copied or transferred, it may be truncated -- "
            f"check its size,\n  or rebuild it with generate_dataset.py then "
            f"train_model.py.\n"
        )