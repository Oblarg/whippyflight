# whippyflight-design

Gain design for [whippyflight-control](https://crates.io/crates/whippyflight-control): structured LQR gains over a grid of motor speeds from a plant fit's coefficient JSON, packed into a `.wfg` blob, and the Pareto front of the LQR weights.

```sh
whippyflight-design gen FIT.json gains.wfg [options]   # one blob
whippyflight-design front FIT.json OUT_DIR [options]   # a front of blobs and its front.json index (--yaw for the yaw weights)
```

- `fit`: the plant-fit coefficient JSON.
- `plant`: the rate and motor-speed model linearized at a motor speed, discretized and augmented with the integral and derivative states.
- `lqr`: excursion weights and the structured LQR solve.
- `blob`: the gain grid, feedforward reference model, integral limit and `.wfg` packing.
- `hover`: the linearized hover loop and its measures (disturbance kick, delay margin, trim, roughness under recorded noise).
- `front`: the Pareto front of the rate and integral weights for pitch/roll or yaw.
- `par`: parallel map with progress and cancellation.

Axes and units follow whippyflight-control (FRD, rot/s, motors by firmware index). Coefficient JSONs and noise recordings come from a firmware's fit tools, such as `whippyflight-betaflight`.

## License

MIT or Apache-2.0, at your option.
