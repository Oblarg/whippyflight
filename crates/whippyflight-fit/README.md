# whippyflight-fit

The plant fit for [whippyflight](https://github.com/Oblarg/whippyflight) from Betaflight blackbox logs: log decoding, the rate and motor-model regressions, fit diagnostics, and the sensor noise the sims replay. Fits are written in whippyflight's canonical frame, ready for [whippyflight-design](https://crates.io/crates/whippyflight-design).

```sh
whippyflight-fit flights LOG.bbl -o flights.json                     # index a log's flights
whippyflight-fit fit LOG.bbl -o FIT.json --dump cli_dump.txt --diagnostics DIAG.json
whippyflight-fit noise LOG.bbl noise.bin --json noise.json --fit FIT.json --diagnostics DIAG.json
whippyflight-fit validate FIT.json LOG.bbl                           # per-flight R², the fit's and each flight's own
whippyflight-fit crossval LOG.bbl --split 3                          # leave-one-out
whippyflight-fit speeds LOG.bbl -o speeds.bin                        # gated motor speeds, for plot_occupancy
```

LOG is a blackbox file (every usable log in it) or `LOG#n` (its n-th, 1-based).

- `bbl`: the blackbox decoder (high-resolution logs rescaled).
- `flight`: a log's signals, typed, and its gating: ground and impacts out.
- `rows`: the regressions' rows, excitation-gated.
- `fit`: the arm ratios, the drag terms' choice, the rate regressions, the motor model by output error, and the voltage-to-telemetry and voltage-to-airframe delays.
- `airframe`, `frame`: the airframe block from the logs' headers and the CLI dump, and the canonical frame.
- `diagnostics`: held-out R², noise splits, turbulence, excerpts, per-flight checks and warnings (`whippyflight_betaflight.fit_diagnostics` plots them).
- `noise`: the sensor noise left once the fit's motion is out, high-passed at the noise splits, and how motor-speed noise grows with speed.
- `scan`, `occupancy`: the flight index and the gated speeds export.
- `run`: the fit as the CLI and the Configurator run it.

The motor model is fitted by output error: its coefficients and the voltage-to-telemetry delay are those whose simulated speeds best match the eRPM telemetry. The CLI dump (`dump` in the Betaflight CLI) records the mixer and motor direction; without it the fit assumes props-in quad X.

## License

MIT or Apache-2.0, at your option.
