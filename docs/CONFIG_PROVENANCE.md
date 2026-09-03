# Where `config/kronos.yaml` came from

**Status: VERIFIED against `PreCDR.ork` (OpenRocket 24.12) on 2026-09-03.**

Every geometry and recovery value below was read directly out of the .ork file.
Values OpenRocket *computes* internally (inertias, drag coefficients, centre of
mass) and the motor internals (referenced from a motor database rather than
stored in the file) are still estimates and are listed as such in §3.

An earlier version of this document said the config was unverified. That was
correct at the time — the .ork was not available. It is now.

---

## 1. Verified — read straight from the .ork

| Parameter | Config | .ork | |
|---|---|---|---|
| `rocket.radius` | 0.078 m | 0.078 m | ✅ |
| `rocket.total_length` | 2.40 m | 2.40 m (sum of components) | ✅ |
| `recovery.drogue.diameter` | 0.60 m | 0.6 | ✅ |
| `recovery.drogue.cd` | 1.5 | 1.5 | ✅ |
| `recovery.drogue.deploy` | apogee | `deployevent=apogee` | ✅ |
| `recovery.main.diameter` | 2.00 m | 2.0 | ✅ |
| `recovery.main.cd` | 2.5 | 2.5 | ✅ |
| `recovery.main.deploy_altitude_agl` | 457.2 m | 457.2 (`deployevent=altitude`) | ✅ |
| `rail.length` | 5.5 m | `launchrodlength=5.5` | ✅ |
| `rail.heading` | 90° | `launchroddirection=90.0` | ✅ |
| `motor.designation` | M1845NT | AeroTech M1845NT | ✅ |

**All five recovery parameters — the ones that actually set the landing point —
check out exactly.** That is the important result.

### The 1500 ft vs 1200 ft question is settled

The .ork deploys the main at **457.2 m AGL (1500 ft)** on an `altitude` event.
Confirmed. If the CDR text's 1200 ft is what the team actually intends, change
`deploy_altitude_agl` to `365.76` — it is worth roughly **91 m of extra drift**
under the main, which is larger than the model's whole main-phase accuracy. This
is now a team decision, not an unknown.

---

## 2. Corrected — the config was wrong, now fixed

| Parameter | Was | Now (.ork) | Note |
|---|---|---|---|
| `nose_cone.length` | 0.60 m | **0.35 m** | .ork nose cone is 0.35 m |
| `fins.span` | 0.13 m | **0.17 m** | from fin points |
| `fins.root_chord` | 0.30 m | **0.29 m** | from fin points |
| `fins.tip_chord` | 0.12 m | **0.14 m** | from fin points |
| `fins.sweep_length` | 0.18 m | **0.20 m** | from fin points |
| `fins.position` | 3.6 m | **1.97 m** | 3.6 m was **off the vehicle** — the airframe is only 2.40 m long |
| `rail.inclination` | 85° | **90°** | `launchrodangle=0.0` (vertical) |
| `monte_carlo.rail_inclination_range` | 83–89° | **84–90°** | now brackets the .ork's vertical rail |

The .ork fin set is a **freeform** set with points
`(0, 0) (0.20, 0.17) (0.34, 0.17) (0.29, 0)`, 4 fins, 5.5 mm airfoil section.
The nose is `shape=haack, shapeparameter=0.0` — a Haack series with C = 0 is
exactly the Von Kármán ogive, so `kind: vonKarman` was already right.

None of these corrections move the landing point much (the 3-DOF simulator does
not read fin or nose geometry at all, and even RocketPy uses them for ascent
aerodynamics rather than descent). They matter because a config that says the
fins are mounted a metre behind the end of the rocket is not one anyone should
be asked to trust.

---

## 3. Corroborated — `dry_mass`

`dry_mass: 15.0 kg` was originally **back-solved** to hit an 11,000 ft apogee,
which is a weak justification. Summing the .ork independently:

| Source | Mass |
|---|---|
| Payload (override) | 3.00 kg |
| Avionics bay (override) | 2.50 kg |
| Airbrakes (override) | 3.00 kg |
| Main + drogue chutes | 0.91 kg |
| Declared mass components | 0.56 kg |
| **Declared subtotal** | **9.86 kg** |
| Airframe structure (geometry × .ork material densities: fibreglass tubes, carbon fins) | ~4.99 kg |
| **Total** | **~14.85 kg** |

This *excludes* couplers, centring rings, bulkheads, shock cords and hardware,
so the true figure is somewhat above 14.85 kg. **15.0 kg is consistent** — the
back-solved number and the component sum agree to about 1%. Still replace it
with the weighed mass once the vehicle exists.

---

## 4. Still estimates — not in the .ork to check against

| Parameter | Value | Why it cannot be verified from the file |
|---|---|---|
| `rocket.inertia_i` / `inertia_z` | 27.0 / 0.15 kg·m² | OpenRocket computes these; not stored |
| `rocket.drag_coefficient` | 0.5 / 0.55 | OpenRocket computes Cd vs Mach; a constant here |
| `rocket.center_of_mass_without_motor` | 1.9 m | Computed by OpenRocket, not stored |
| `motor.average_thrust` | 1845 N | From the designation M**1845**NT, not a thrust curve |
| `motor.burn_time` | 4.5 s | Arithmetic: 8308 ÷ 1845 |
| `motor.propellant_mass` | 4.2 kg | Arithmetic: 8308 ÷ (200 s × g) |
| `motor` grain/nozzle geometry | various | The .ork references the motor from a database |

To close these: open the .ork in OpenRocket and read the computed CG/CP and
stability off the UI, and pull the real M1845NT thrust curve from ThrustCurve.org.
The descent — which sets the landing point — is insensitive to all of them.

---

## 5. Deliberate deviations from the .ork

| Parameter | .ork | Config | Why |
|---|---|---|---|
| `launch_site` | 28.61, −80.60 (Florida) | −30.8506, 143.0847 (White Cliffs) | The .ork ships OpenRocket's Florida default. The competition is at White Cliffs, and live lat/lon output has to be real. |
| `environment.wind_speed` | 5.0 m/s (`windaverage`) | 6.0 nominal, 1–12 Monte-Carlo | The model trains across a wind *distribution*; a single nominal would defeat the point. |

---

## 6. Remaining actions

1. Confirm **1500 ft vs 1200 ft** main deploy with the recovery lead (§1).
2. Replace `dry_mass` with the **weighed** mass when the vehicle is built, then
   regenerate and retrain.
3. Read CG / CP / stability out of OpenRocket and update `center_of_mass_without_motor`.
4. Pull the real M1845NT thrust curve if you want trustworthy apogee numbers
   (it does not affect the landing prediction).
5. Install RocketPy before flight so the surrogate is trained on 6-DOF physics
   rather than the built-in 3-DOF fallback. `scripts/check_model.py` reports
   which engine trained the model you are about to fly.
