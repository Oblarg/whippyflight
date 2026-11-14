# whippyflight

A gain-scheduled LQR attitude-rate controller for multirotors, independent
of any one flight firmware.  The gains are solved offline from a plant
model fitted to flight logs, over a grid of motor speeds, and packed into a
gain blob (`.wfg`) the controller loads at boot.  A small adapter per
firmware connects the controller to that firmware's sensors, mixer and
storage.

| Path | What |
|------|------|
| `crates/whippyflight-control` | The controller: `no_std`, no allocation, no globals.  Loads blobs, runs the control law. |
| `crates/whippyflight-sim` | Nonlinear plant model, stick-flick and delay-margin evaluations, sensor-noise replay. |
| `python/whippyflight` | Gain generation (`gen_wfg`), Pareto fronts (`front_wfg`), the coefficient frame (`coefficients`), flick plots. |
| `adapters/betaflight` | Betaflight's tools: blackbox decoding, the plant fit, noise extraction (`python/whippyflight_betaflight`), and the SITL test harness (`tools/`). |
| `fixtures` | Fitted airframes, their noise recordings, the committed test blob, and the eval baseline. |
| `tools` | Development: the test pipeline, LQR weight sweeps, the venv's requirements. |

The Betaflight adapter crate itself, the C ABI the firmware calls, lives in
the Betaflight tree (`src/main/flight/whippyflight`), under Betaflight's
license.

## The contract

Everything a firmware adapter and a fit tool must agree on.

### Units

Rates and motor shaft speeds in rot/s, voltages in V, specific force in
m/s², time in s.  Motor output is a duty fraction in [0, 1], throttle a
fraction in [0, 1].  An adapter converts its firmware's units on the way in
and out (`whippyflight_control::types`).

### Axes: FRD

Every 3-axis value is in the body's FRD frame (x forward, y right, z down;
right-handed), ordered `[roll, pitch, yaw]`:

- positive roll lowers the right side,
- positive pitch raises the nose,
- positive yaw turns the nose right,
- level and at rest, the accelerometer reads −g₀ on z.

A firmware in another frame converts at its boundary.  Betaflight's body
frame is FLU (x forward, y left, z up), so its adapter negates pitch and yaw
(and the accelerometer's y and z) in both directions.

### Motors: the firmware's own indices

Every 4-motor array is by the firmware's motor index, in whatever order its
mixer numbers them.  The blob records where each index sits and which way
it spins (its layout block: a position code per motor, front-left 0,
rear-left 1, front-right 2, rear-right 3, then a spin per motor, 0 CW and
1 CCW, viewed from above).  The loader rejects a layout that isn't a quad X
(`BadLayout`), and an adapter must reject one its mixer doesn't drive
(`LayoutMismatch`), so gains never fly on motors they weren't solved for.
Only quad X is supported.

### The gain blob

Format version 9, documented field by field in
`crates/whippyflight-control/src/gains.rs`: a 32-byte header (magic `LQRG`,
version, grid resolution N, speed range, loop period, filter time
constants, total size), fixed-size parameter blocks ending in the layout
block, the gain tables quantized to `u16` over an N⁴ motor-speed grid, and a
CRC32.  At N = 4 a blob is 25 856 bytes.  `load_gains` validates every
field before it touches the storage, and returns a `GainLoadResult`.

The controller reads the tables in place, so the adapter keeps the blob in
static, 4-byte-aligned memory until it calls for the next load.

### Coefficient JSON

The plant fit a firmware's tools write and `gen_wfg` reads.  It is in the
same frame, and its `frame` block says how the firmware numbers its motors:

```json
"frame": {
  "axes": "FRD",
  "motors": [
    {"position": "rear_right", "spin": "ccw"},
    {"position": "front_right", "spin": "cw"},
    {"position": "rear_left", "spin": "cw"},
    {"position": "front_left", "spin": "ccw"}
  ]
}
```

with one entry per firmware motor index.  The arms (`pitch.dFront`,
`pitch.dBack`, `roll.dLeft`, `roll.dRight`) are physical, and the drag
coefficients map FRD specific force to FRD angular acceleration.  Fits
without a `frame` block are Betaflight fits from before it existed;
`python -m whippyflight.coefficients FIT.json --write OUT.json` converts
one.

## Writing an adapter

An adapter is the controller's HAL.  Per loop it hands
`ControllerState::update_with_target` the gyro, motor speeds, battery
voltage and accelerometer (`SensorData`), the throttle, and the setpoint
estimator's target, and writes the returned duties to its motors.  It also:

- converts its units and axes to the contract's, and telemetry back to its own,
- stores the blob and loads it with `gains::load_gains`, checking
  `GainStorage::layout` against its mixer,
- feeds RX packets to `setpoint::SetpointEstimator`,
- resets the controller on disarm.

The Betaflight adapter (`src/main/flight/whippyflight` in the Betaflight
tree: `bf.rs` for the conventions, `ffi.rs` for the C ABI) is the
reference.  Its firmware side builds it with cargo when the firmware is
built with `OPTIONS=USE_WHIPPYFLIGHT`.

A firmware's fit tools write the coefficient JSON above.  `gen_wfg` and the
evaluations don't care which firmware the fit came from.

## Python

Python 3.11 or newer.  The development venv installs both packages,
editable, at the versions the eval baseline was recorded with:

```sh
python3 -m venv tools/.venv
tools/.venv/bin/pip install -r tools/requirements.txt
```

Gains from a fit:

```sh
tools/.venv/bin/python -m whippyflight.gen_wfg FIT.json gains.wfg \
    --dt 1.25e-4 --resolution 4 --idle-rps 180
```

A Betaflight fit from blackbox logs, with the CLI dump that records the
mixer and motor direction:

```sh
tools/.venv/bin/python -m whippyflight_betaflight.fit_model fit LOG.bbl \
    -o FIT.json --dump cli_dump.txt
```

The Betaflight Configurator's WhippyFlight tab runs these, and the noise
extraction and plots, as modules of the interpreter it is pointed at.

## Testing

```sh
tools/.venv/bin/python tools/pipeline.py check   # committed blob up to date with its fit
tools/.venv/bin/python tools/pipeline.py test    # cargo test, the whole workspace
tools/.venv/bin/python tools/pipeline.py eval    # evals per airframe, against the baseline
```

`eval --save` records a new baseline.  The Betaflight SITL test flies the
committed blob through the firmware's SITL build:

```sh
tools/.venv/bin/python adapters/betaflight/tools/sitl.py
```

with the Betaflight checkout in `$BETAFLIGHT_DIR` (or next to this one).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.  Firmware adapters inside a
firmware's tree take that firmware's license.
