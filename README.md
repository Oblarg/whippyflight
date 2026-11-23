# whippyflight

A gain-scheduled LQR attitude-rate controller for quadrotors.

| Path | What |
|------|------|
| `control-law.tex` | Derivation of the attitude rate control law. |
| `crates/whippyflight-control` | The controller: `no_std`, no allocation, no globals. |
| `crates/whippyflight-sim` | Nonlinear plant, flick and delay-margin evaluations, sensor-noise replay. |
| `crates/whippyflight-design` | Gain design: structured LQR, gain blobs, Pareto fronts of the LQR weights. |
| `python/whippyflight` | Gain generation (`gen_wfg`), coefficient frames, flick and gain plots. |
| `adapters/betaflight` | Betaflight tools: blackbox decoding, plant fit, noise extraction, SITL harness. |
| `fixtures` | Fitted airframes, noise recordings, the test blob, the eval baseline. |
| `tools` | Test pipeline, LQR weight sweeps, venv requirements. |

## Conventions

**Units.** Rates and motor shaft speeds in rot/s, voltages in V, specific force in m/s², time in s. Throttle is a fraction in [0, 1].

**Voltage and duty.** The controller works in volts but outputs duty in [0, 1], applied voltage over `SensorData::battery_voltage`, since battery voltage is the ceiling it saturates against. Adapters supply the battery voltage their logs record and map duty onto their ESC protocol.

**Axes.** FRD body frame (x forward, y right, z down), ordered `[roll, pitch, yaw]`: positive roll lowers the right side, positive pitch raises the nose, positive yaw turns the nose right. At rest the accelerometer reads −g₀ on z.

**Motors.** Arrays are by the firmware's motor index. The gain blob's layout block gives each index a position (front-left 0, rear-left 1, front-right 2, rear-right 3) and spin (0 CW, 1 CCW, from above). The loader rejects non-quad-X layouts (`BadLayout`); an adapter rejects layouts its mixer doesn't drive (`LayoutMismatch`). Only quad X is supported.

**Gain blob (`.wfg`).** A 32-byte header, parameter blocks ending in the layout, `u16`-quantized gain tables over an N⁴ motor-speed grid, and a CRC32; the format is documented in `crates/whippyflight-control/src/gains.rs`. `load_gains` validates it and returns a `GainLoadResult`. Tables are read in place, so the blob must stay in static, 4-byte-aligned memory while loaded.

**Coefficient JSON.** The plant fit `gen_wfg` reads, in the same axes. Arms (`pitch.dFront`/`dBack`, `roll.dLeft`/`dRight`) are physical, and a `frame` block lists each firmware motor in index order:

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

## Adapters

An adapter is the controller's HAL. Each loop it passes sensor data (`SensorData`), throttle and the setpoint target to `ControllerState::update_with_target` and writes out the returned duties. It also converts units and axes both ways, stores the blob and loads it with `gains::load_gains` (checking `GainStorage::layout` against its mixer), feeds RX packets to `setpoint::SetpointEstimator`, and resets the controller on disarm.

The Betaflight adapter (`src/main/flight/whippyflight` in the Betaflight tree) is the reference; build the firmware with `OPTIONS=USE_WHIPPYFLIGHT`.

## Python

Python 3.11+, from the repo root:

```sh
python3 -m venv tools/.venv
tools/.venv/bin/pip install -r tools/requirements.txt
```

Fit a Betaflight log (the CLI dump records mixer and motor direction), then generate gains:

```sh
tools/.venv/bin/python -m whippyflight_betaflight.fit_model fit LOG.bbl \
    -o FIT.json --dump cli_dump.txt
tools/.venv/bin/python -m whippyflight.gen_wfg FIT.json gains.wfg \
    --dt 1.25e-4 --resolution 4 --idle-rps 180
```

The Betaflight Configurator's WhippyFlight tab runs the same pipeline.

## Testing

```sh
tools/.venv/bin/python tools/pipeline.py check   # test blob up to date with its fit
tools/.venv/bin/python tools/pipeline.py test    # cargo test
tools/.venv/bin/python tools/pipeline.py eval    # evals against the baseline (--save to update)
tools/.venv/bin/python adapters/betaflight/tools/sitl.py   # Betaflight SITL, $BETAFLIGHT_DIR
```

## License

MIT or Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)). Adapters inside a firmware's tree take that firmware's license.
