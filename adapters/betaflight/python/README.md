# whippyflight-betaflight

Betaflight tools for [whippyflight](https://github.com/Oblarg/whippyflight): blackbox decoding, the plant fit, and sensor-noise extraction. Fits are written in whippyflight's canonical frame, ready for `whippyflight.gen_wfg`.

```sh
python -m whippyflight_betaflight.fit_model flights LOG.bbl -o flights.json      # index a log's flights
python -m whippyflight_betaflight.fit_model fit LOG.bbl -o FIT.json --dump cli_dump.txt --diagnostics DIAG.json
python -m whippyflight_betaflight.fit_diagnostics DIAG.json OUT_DIR              # fit plots (extra: plots)
python -m whippyflight_betaflight.extract_noise LOG.bbl noise.bin --fit FIT.json --diagnostics DIAG.json
```

The diagnostics place each sensor's noise split, where the fit's residual turns from the airframe's to the sensor's; noise extraction high-passes there, after taking out the motion the fit explains, and the turbulence estimate stops there. Without them extraction falls back to fixed cut-offs. With the fit it also measures how the motor-speed noise grows with motor speed over the whole flight, and `--json` records it for the sim.

The motor model is fitted by output error: its coefficients and the voltage-to-telemetry delay are those whose simulated speeds best match the eRPM telemetry. The fit records one loop tick of that delay as the motors' and the rest as the telemetry's; `--select r` keeps main.R's regression and records none.

The CLI dump (`dump` in the Betaflight CLI) records the mixer and motor direction; without it the fit assumes props-in quad X.

## License

MIT or Apache-2.0, at your option.
