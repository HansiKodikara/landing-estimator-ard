# Kronos Landing-Zone Estimator (LZE)

A **live recovery assistant** for the ARD *Project Kronos* rocket (AURC 2026). It
watches the downlinked telemetry and continuously predicts **where the rocket
will land**, drawing a likely recovery area that tightens as the flight
progresses — so the ground team can dispatch a vehicle *before* touchdown across
the vast, mobile-dead terrain of White Cliffs, NSW.

```
Rocket sends telemetry while flying
        ↓
Ground station / Raspberry Pi receives it   (915 MHz LoRa, 1 Hz GPS)
        ↓
System estimates the rocket's current state (lze.estimator)
        ↓
Surrogate model predicts remaining drift    (lze.model — learned from RocketPy)
        ↓
Screen shows a live landing zone            (lze.live + dashboard/)
```

The model is a **surrogate**: a small, fast approximation of the RocketPy physics
simulator. Instead of the Raspberry Pi running a full simulation every second, it
runs a learned model (sub-millisecond) that was trained offline on thousands of
RocketPy-simulated states.

Crucially it does **not** predict `launch site → landing site`. It predicts:

```
current rocket state  →  remaining (east, north) offset to landing
```

> *"From where the rocket is right now, how much farther will it drift before it
> lands?"*

That is why the prediction **improves in real time**: early in the flight a lot
can still happen, but near apogee there is more flight history, and under
parachute the descent rate and wind drift become clear — so the recovery area
collapses toward a tight circle.

---

## Measured performance

Committed model: 80 RocketPy Monte-Carlo flights of the Kronos vehicle, evaluated
live (noisy 1 Hz telemetry → estimator → surrogate → smoothing) on **fresh**
held-out flights. Mean landing-point error by phase:

| Phase        | When                    | Mean landing error |
|--------------|-------------------------|--------------------|
| boost        | on the rail / burning   | ~710 m             |
| coast        | up to apogee            | ~870 m *(peak uncertainty)* |
| **drogue**   | descending under drogue | **~170 m**         |
| **main**     | under main chute        | **~44 m**          |
| final fix    | at touchdown            | **~38 m (median)** |

The recovery area is honestly wide near apogee and tightens by **~20×** under the
parachutes. Reproduce with `python scripts/evaluate_live.py`.

---

## Quick start

```bash
pip install -r requirements.txt          # numpy, sklearn, rocketpy, ...

# 1. Generate training data from RocketPy (or the built-in fallback sim)
python scripts/generate_dataset.py --n-flights 80 --out data/dataset.npz

# 2. Train the surrogate (leak-free, split by whole flight)
python scripts/train_model.py --dataset data/dataset.npz --out data/surrogate.joblib

# 3a. End-to-end demo -> prints the prediction sharpening + writes a shareable,
#     self-contained replay dashboard (open in any browser, no server/internet)
python scripts/demo.py --out flight_replay.html

# 3b. Or run the LIVE dashboard server (replay demo, accelerated real time)
python scripts/run_live.py --port 8000        # then open http://127.0.0.1:8000
```

Steps 1–2 build `data/dataset.npz` and `data/surrogate.joblib` (a few minutes
with RocketPy; instant with `--engine fallback`). These generated artifacts are
git-ignored — run them once before the demo, or point the scripts at your own
paths.

### Deploying the model to the ground station

Training happens once on a laptop; the Raspberry Pi only ever *loads* the
result. `scripts/train_model.py` writes a single portable artifact
(`data/surrogate.joblib`) holding the trained trees plus provenance -- the
feature contract, evaluation scores, and the library versions it was built
with. Copy that one file to the Pi (it is git-ignored on purpose: it is a
regenerable binary), then verify it *on the Pi* before you drive out:

```bash
python scripts/check_model.py --model data/surrogate.joblib
# -> PASS - safe to fly     (exit 0; exit 1 means do not fly)
```

The check confirms the file loads, that its feature contract matches the code,
how it compares to the training environment, that it predicts sanely on a
reference state, and that inference fits inside the telemetry budget. Loading a
model trained on a different feature set is a hard error rather than a silent
wrong answer; a library-version difference is only a warning. To prove the Pi
reproduces the training machine bit-for-bit, pass the numbers it printed:

```bash
python scripts/check_model.py --expect 366.3,-154.5
```

### On the ground station (Raspberry Pi)

```bash
pip install -r requirements.txt pyserial
python scripts/run_live.py --source serial --serial-port /dev/ttyUSB0 --host 0.0.0.0
```

The flight computer's telemetry board forwards each decoded LoRa frame as one
line of JSON (`TelemetryPacket` schema); the dashboard is served locally and
needs no internet — matching the CDR's "local server, no internet dependency".

### Alongside the ARD dashboard

The LZE can run **next to the ARD Ground-Station Dashboard** (a separate repo)
without any change to it: point the recovery map at ARD's telemetry and it
serves its own offline landing view.

```bash
# A) push stream over Socket.IO -- lowest latency, every frame
pip install '.[ard]'                     # optional Socket.IO client
python scripts/run_live.py --source ard --ard-url http://127.0.0.1:5000 --port 8000

# B) ARD's documented REST API -- no websocket, no extra dependencies.
#    Backfills GET /telemetry/history, then polls GET /telemetry/latest.
python scripts/run_live.py --source ard-rest --ard-url http://127.0.0.1:5000 --ard-poll-hz 4

# C) replay a captured ARD telemetry log offline (no network / deps)
python scripts/run_live.py --source ard-file --ard-file capture.jsonl
```

Prefer (A): polling at 4 Hz against 10 Hz telemetry sees roughly every third
frame. That costs the landing prediction little -- the estimator is driven by
GPS-rate motion, not packet count -- but (B) exists for when the websocket is
awkward (CORS, a proxy, or not wanting `python-socketio` on the Pi).

`lze.telemetry.ard_adapter` converts ARD's telemetry envelope into the
`TelemetryPacket` schema (reconstructing the ENU velocity vector from successive
fixes). See `docs/REPORT.md` §3.

### Optional launch-day wind forecast

By design the model is **any-weather**: it trains over a wind distribution and
infers the real wind live from parachute drift, so it needs no day-of retrain.
A forecast can *optionally* seed the early (boost/coast) predictions — train a
seeded model and pass the forecast at inference:

```bash
python scripts/generate_dataset.py --n-flights 80 --forecast-wind-noise 2.5 --out data/dataset_fc.npz
python scripts/train_model.py --dataset data/dataset_fc.npz --out data/surrogate_fc.joblib
python scripts/run_live.py --model data/surrogate_fc.joblib --wind-seed-speed 7 --wind-seed-from 240
```

Seeding is off by default (committed behaviour unchanged). See `docs/REPORT.md` §4.

---

## How it works

### 1. RocketPy generates flight experience — `lze.sim`
`rocketpy_sim.py` builds the Kronos vehicle from `config/kronos.yaml` (AeroTech
M1845NT motor, 156 mm airframe, drogue `Cd·⌀` 1.5/0.6 m at apogee, main 2.5/2.0 m
at 457 m AGL — all from the CDR and the OpenRocket file) and flies it. Mass is
calibrated so apogee ≈ 11,000 ft, the CDR's simulated target. `fallback.py` is a
pure-Python 3-DOF integrator used automatically when RocketPy isn't installed, so
the whole stack still runs on a bare Pi or in CI.

### 2. The surrogate learns from those simulations — `lze.model`
`scripts/generate_dataset.py` runs a Monte-Carlo over wind, rail angle, mass and
drag. Every flight is turned into supervised rows of *`current state → remaining
offset`* — but the state features are produced by running the **same online
estimator** the live system uses, over **noisy** replayed telemetry. This
"train like you infer" step removes train/serve skew: the model only ever sees
information that will actually be available in flight (e.g. wind is unknown until
the chutes open). The surrogate is three `HistGradientBoostingRegressor`s (for
`rem_e`, `rem_n`, `rem_t`) — tiny to store, sub-ms to evaluate.

### 3. The Raspberry Pi estimates the live state — `lze.estimator`
`OnlineStateEstimator` fuses GPS + barometric altitude, follows the CDR flight
state machine (boost → coast → apogee → drogue → main), and **infers wind** from
horizontal motion: under a parachute the airframe drifts with the air mass, so
its horizontal velocity *is* the wind. That inferred wind feeds the surrogate,
which is why accuracy jumps once the chutes deploy.

### 4. The landing zone updates live — `lze.live`
`LandingPredictor` chains estimator → surrogate → geodesy: predicted landing =
current GPS position + predicted remaining offset, converted back to lat/lon. A
short EMA smooths per-frame model jitter into a stable point that glides toward
the true landing. The **uncertainty radius** shown on the map is the model's own
held-out error for the current phase, so the drawn recovery area is calibrated,
not decorative. `server.py` streams predictions to the browser dashboard over
Server-Sent Events; `dashboard/index.html` is a self-contained, offline canvas
map (no external tiles/libraries).

---

## Repository layout

```
config/kronos.yaml          Vehicle / site / recovery parameters (from CDR + .ork)
src/lze/
  config.py, geo.py         Config loader, ENU <-> lat/lon conversions
  sim/                      RocketPy model, 3-DOF fallback, Monte-Carlo dataset
  model/                    Feature engineering, training, surrogate wrapper
  estimator/                Online state + wind + phase estimator
  telemetry/                Packet schema, replay stream, serial/UDP sources
  live/                     Predictor, SSE server, baked replay-page generator
dashboard/index.html        Offline live landing-zone dashboard
scripts/                    generate_dataset · train_model · evaluate_live · demo · run_live
tests/                      pytest suite (runs on the fallback engine, no GPU/net)
data/                       Committed dataset + trained surrogate
```

## Notes & assumptions
- Launch site defaults to White Cliffs, NSW (CDR). The OpenRocket file ships a
  Florida default; override `launch_site` in the config for other sites.
- Main-chute deploy altitude follows the `.ork` (457 m / 1500 ft AGL). The CDR
  text mentions 1200 ft — change `recovery.main.deploy_altitude_agl` if needed.
- The M1845NT thrust curve is approximated as constant average thrust matching
  the published 8308 N·s total impulse; the descent (which sets the landing
  point) is insensitive to boost thrust shape.
- Telemetry noise, GPS/baro rates and Monte-Carlo ranges are all in
  `config/kronos.yaml`.
