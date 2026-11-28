# whippyflight-betaflight

Betaflight tools for [whippyflight](https://github.com/Oblarg/whippyflight): blackbox decoding, fit plots, motor-speed occupancy, and a motor's roughness by source. The fit, its diagnostics and sensor-noise extraction are [whippyflight-fit](https://crates.io/crates/whippyflight-fit)'s.

```sh
python -m whippyflight_betaflight.fit_diagnostics DIAG.json OUT_DIR              # fit plots (extra: plots)
python -m whippyflight_betaflight.plot_occupancy gains.wfg OUT_DIR speeds.bin    # flight time over motor-speed space (extra: plots)
python -m whippyflight_betaflight.effort LOG.bbl                                 # a motor's roughness by source
```

`speeds.bin` is `whippyflight-fit speeds LOG.bbl... -o speeds.bin`.

`effort` reads a flight logged with `debug_mode = WHIPPYFLIGHT`, which logs one motor's (`whippyflight_debug_motor`) feedback by term, feedforward, affine linearizer, gain-schedule jitter, applied voltage and filtered speed, and reports each term's share of the applied voltage's roughness.

Under WhippyFlight, blackbox's PID and setpoint fields carry the controller's own values: `axisP`, `axisI` and `axisD` its rate error, integral state over its time constant and derivative state; `axisF` the feedforward target (0.1 deg/s); `setpoint[0..2]` the tracked setpoint and `setpoint[3]` the allocator's command scale (‰); `rcCommand[0..2]` the raw RX setpoint (0.1 deg/s), with throttle still in `rcCommand[3]`.

`plot_occupancy` shows where a flight spends its time in a blob's motor-speed space, and how often the feedback and feedforward gains scheduled there differ from the same gains frozen at hover, a fixed PID behind a mixer, and how much voltage the affine linearizer contributes there.

## License

MIT or Apache-2.0, at your option.
