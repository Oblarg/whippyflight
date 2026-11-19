# whippyflight

Gain generation for [whippyflight-control](https://crates.io/crates/whippyflight-control): LQR gains solved (DARE) over a grid of motor speeds from a plant fit's coefficient JSON, packed into a `.wfg` blob.

```sh
python -m whippyflight.gen_wfg FIT.json gains.wfg --dt 1.25e-4 --resolution 4 --idle-rps 180
python -m whippyflight.front_wfg FIT.json OUT_DIR      # a Pareto front of blobs over the LQR weights
python -m whippyflight.coefficients FIT.json           # a coefficient JSON in the canonical frame
python -m whippyflight.plot_flicks flicks.json OUT_DIR # plots of the Configurator's flick report (extra: plots)
python -m whippyflight.plot_gains gains.wfg OUT_DIR    # a blob's gains over motor-speed space (extra: plots)
```

Coefficient JSONs come from a firmware's fit tools, such as `whippyflight-betaflight`. Axes, units and motor layout are documented in the [whippyflight](https://github.com/Oblarg/whippyflight) repo.

## License

MIT or Apache-2.0, at your option.
