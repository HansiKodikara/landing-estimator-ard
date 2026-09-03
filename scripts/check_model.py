#!/usr/bin/env python3
"""Pre-flight check for a trained surrogate. Run this ON THE GROUND STATION.

Verifies that the model file you carried to the launch site actually loads and
predicts correctly on *this* machine, before anyone drives into the desert:

    python scripts/check_model.py --model data/surrogate.joblib

Checks, in order:
  1. the file exists and loads
  2. its feature contract matches this code   (fatal if not)
  3. the training environment vs this one     (warning if different)
  4. it predicts sanely on a known reference state
  5. inference is fast enough for the telemetry rate

Exit code 0 = safe to fly, 1 = do not fly. Add ``--expect`` on the ground
station to assert bit-identical output against the machine that trained it.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from lze.model.surrogate import REFERENCE_STATE, Surrogate

OK, BAD, WARN = "  [ok]", "  [FAIL]", "  [warn]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="data/surrogate.joblib")
    ap.add_argument("--rate-hz", type=float, default=1.0,
                    help="Telemetry rate the Pi must keep up with")
    ap.add_argument("--expect", default=None,
                    help="Expected 'rem_e,rem_n' from the training machine, to "
                         "confirm identical output across machines")
    args = ap.parse_args()
    problems = 0

    print(f"Model file : {args.model}")
    path = Path(args.model)
    if not path.exists():
        print(f"{BAD} file not found. Copy it from the machine that ran "
              f"scripts/train_model.py -- it is deliberately not in git.")
        return 1
    print(f"{OK} found ({path.stat().st_size / 1e6:.1f} MB)")

    # --- load (raises on a feature-contract mismatch) ---
    try:
        sur = Surrogate.load(args.model)
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} could not load: {exc}")
        return 1
    print(f"{OK} loaded and feature contract matches this code")

    # --- provenance ---
    env = sur.metadata.get("env") or {}
    if env:
        print(f"\nTrained {env.get('saved_utc', '?')} with "
              f"{env.get('model_class', '?')}")
        print(f"  training env : python {env.get('python','?')}, "
              f"sklearn {env.get('sklearn','?')}, numpy {env.get('numpy','?')}")
    print(f"  this machine : python {platform.python_version()} on "
          f"{platform.machine()}")
    for w in sur.check_compatibility():
        print(f"{WARN} {w}")

    # Which simulator taught it? The fallback is far cruder than RocketPy and
    # the difference is invisible once the model is a binary blob.
    engine = sur.metadata.get("engine")
    if engine is None:
        print(f"{WARN} model does not record its training engine (built before "
              f"provenance stamping) -- retrain to identify it")
    elif "rocketpy" in str(engine).lower():
        print(f"{OK} trained by RocketPy")
    else:
        print(f"{WARN} trained by '{engine}', NOT RocketPy. This is the built-in "
              f"3-DOF approximation -- fine for testing, but expect degraded "
              f"real-world accuracy. Install rocketpy and regenerate for flight.")

    ev = sur.metadata.get("eval") or {}
    if ev.get("per_phase_mae_m"):
        pp = ev["per_phase_mae_m"]
        print("  held-out error: " + "  ".join(
            f"{k} {v:.0f}m" for k, v in pp.items()))
    else:
        print(f"{BAD} no evaluation metadata -- the recovery-zone radius would "
              f"be a hardcoded guess rather than measured error. Retrain.")
        problems += 1

    # --- prediction on a known state ---
    print("\nReference prediction (mid-descent under main, westerly):")
    try:
        p = sur.predict(REFERENCE_STATE)
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} prediction raised: {exc}")
        return 1
    print(f"  rem_e={p.rem_e:+.1f} m  rem_n={p.rem_n:+.1f} m  rem_t={p.rem_t:.1f} s")

    finite = all(abs(v) < 1e5 for v in (p.rem_e, p.rem_n, p.rem_t))
    if not finite:
        print(f"{BAD} implausible output -- the model is corrupt or mismatched")
        problems += 1
    elif p.rem_t <= 0:
        print(f"{WARN} remaining time <= 0 while still descending; check training data")
    else:
        print(f"{OK} output is finite and plausible")

    if args.expect:
        want_e, want_n = (float(v) for v in args.expect.split(","))
        drift = max(abs(p.rem_e - want_e), abs(p.rem_n - want_n))
        if drift > 0.5:
            print(f"{BAD} differs from the training machine by {drift:.2f} m "
                  f"(expected {want_e:+.1f},{want_n:+.1f}) -- likely a library "
                  f"version difference. Match versions before flying.")
            problems += 1
        else:
            print(f"{OK} matches the training machine to within {drift:.3f} m")

    # --- speed ---
    for _ in range(3):
        sur.predict(REFERENCE_STATE)
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        sur.predict(REFERENCE_STATE)
    ms = (time.perf_counter() - t0) / n * 1000
    budget = 1000.0 / args.rate_hz
    print(f"\nInference: {ms:.1f} ms per frame "
          f"(budget {budget:.0f} ms at {args.rate_hz:g} Hz)")
    if ms > budget:
        print(f"{BAD} too slow for the telemetry rate -- use a smaller model")
        problems += 1
    elif ms > budget * 0.5:
        print(f"{WARN} using over half the budget; little headroom")
    else:
        print(f"{OK} {budget / ms:.0f}x headroom")

    print("\n" + ("PASS - safe to fly" if problems == 0
                  else f"FAIL - {problems} blocking problem(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
