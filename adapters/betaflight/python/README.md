# whippyflight-betaflight

Betaflight tools for [whippyflight](https://github.com/Oblarg/whippyflight): blackbox decoding, the plant fit, and sensor-noise extraction. Fits are written in whippyflight's canonical frame, ready for `whippyflight.gen_wfg`.

```sh
python -m whippyflight_betaflight.fit_model flights LOG.bbl -o flights.json      # index a log's flights
python -m whippyflight_betaflight.fit_model fit LOG.bbl -o FIT.json --dump cli_dump.txt --diagnostics DIAG.json
python -m whippyflight_betaflight.fit_diagnostics DIAG.json OUT_DIR              # fit plots (extra: plots)
python -m whippyflight_betaflight.extract_noise LOG.bbl noise.bin --fit FIT.json --diagnostics DIAG.json
python -m whippyflight_betaflight.plot_occupancy gains.wfg OUT_DIR LOG.bbl...    # flight time over motor-speed space (extra: plots)
python -m whippyflight_betaflight.effort LOG.bbl                                 # a motor's roughness by source
```

`effort` reads a flight logged with `debug_mode = WHIPPYFLIGHT`, which logs one motor's (`whippyflight_debug_motor`) feedback by term, feedforward, affine linearizer, gain-schedule jitter, applied voltage and filtered speed, and reports each term's share of the applied voltage's roughness.

Under WhippyFlight, blackbox's PID and setpoint fields carry the controller's own values: `axisP`, `axisI` and `axisD` its rate error, integral state over its time constant and derivative state; `axisF` the feedforward target (0.1 deg/s); `setpoint[0..2]` the tracked setpoint and `setpoint[3]` the allocator's command scale (‰); `rcCommand[0..2]` the raw RX setpoint (0.1 deg/s), with throttle still in `rcCommand[3]`.

`plot_occupancy` shows where a flight spends its time in a blob's motor-speed space, and how often the feedback and feedforward gains scheduled there differ from the same gains frozen at hover, a fixed PID behind a mixer, and how much voltage the affine linearizer contributes there.

The diagnostics place each sensor's noise split, where the fit's residual turns from the airframe's to the sensor's; noise extraction high-passes there, after taking out the motion the fit explains, and the turbulence estimate stops there. Without them extraction falls back to fixed cut-offs. With the fit it also measures how the motor-speed noise grows with motor speed over the whole flight, and `--json` records it for the sim.

The motor model is fitted by output error: its coefficients and the voltage-to-telemetry delay are those whose simulated speeds best match the eRPM telemetry. The fit records one loop tick of that delay as the motors' and the rest as the telemetry's; `--select r` keeps main.R's regression and records none.

The CLI dump (`dump` in the Betaflight CLI) records the mixer and motor direction; without it the fit assumes props-in quad X.

## License

MIT or Apache-2.0, at your option.
