# Landing Zone Estimator Report

This project is a live recovery assistant for Project Kronos. During flight, it reads telemetry and continuously predicts the rocket landing point, then shows a live recovery zone on a local dashboard.

## Why there is a separate trained model (surrogate)

The system uses two different compute stages:

- Offline training stage (before launch): run many heavier simulations to generate examples.
- Live inference stage (during flight): run a small trained model every telemetry update.

That small trained model is the surrogate model. A surrogate model is a fast approximation of a slower physics simulator: it learns the mapping from current rocket state to remaining drift-to-landing, then predicts it in milliseconds.

Why this matters for Raspberry Pi:

- Re-running full physics simulation each second on a PRaspberry Pi is too expensive for robust real-time use.
- The pre-trained surrogate is lightweight and fast, so the Pi can keep up with live telemetry and update the recovery zone continuously.
- Result: heavy compute happens once offline; fast prediction happens live on-device.

## What this project is

- A real-time landing prediction system for field recovery operations.
- A surrogate-model approach: fast ML inference replaces heavy physics simulation during flight.
- An offline-first tool intended for remote launch sites with limited or no internet.

## What it can achieve currently

1. Live landing-zone prediction
- Consumes telemetry and continuously outputs predicted landing latitude/longitude and uncertainty radius.

2. Prediction tightening during flight
- Early phases are intentionally broad.
- Accuracy improves under drogue and main as wind/descent become observable.

3. End-to-end operational flow
- Can generate training data, train a surrogate, evaluate performance, and serve a live dashboard.
- Supports replay mode and serial telemetry input for ground-station use.

4. ARD dashboard interoperability
- Can run alongside ARD telemetry by adapting ARD-shaped telemetry envelopes into the LZE packet format.
- Provides both live bridge and offline replay ingestion paths.

5. Fully offline runtime mode
- Local server and local dashboard do not require cloud services.
- Fallback simulation path enables operation without RocketPy.

## Current progress status

- Core pipeline is implemented: telemetry -> state estimator -> surrogate -> live map output.
- Main scripts for data generation, training, evaluation, and live serving are in place.
- Automated test coverage exists across geodesy, simulation, estimator behavior, model/live behavior, and ARD integration.
- Project is in a strong MVP state and suitable for demo and iterative field hardening.

## Practical limitations right now

- Early-flight uncertainty remains large by design (before good wind observability).
- Best performance depends on having a trained model artifact prepared before field deployment.
- Additional long-run operational hardening is still a normal next step for production readiness.
