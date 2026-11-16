# whippyflight-sim

Host-side simulation for [whippyflight-control](https://crates.io/crates/whippyflight-control): the nonlinear quadrotor plant its gains are solved for, and the evaluations that fly the controller through it.

- `plant`: the rate and motor-speed model, read from a plant-fit coefficient JSON (`Model::from_json`), integrated with RK4, with the JSON's motor-command and speed-telemetry delays.
- `flick`: stick flicks through the setpoint estimator and controller, with traces and tracking metrics.
- `margin`: delay margin, disturbance-kick recovery, trim and hover roughness.
- `noise`: replay of recorded gyro and motor-speed noise, the motor-speed noise optionally following each motor's speed.

Axes and units follow whippyflight-control (FRD, rot/s, motors by firmware index). Coefficient JSONs and noise recordings come from the [whippyflight](https://github.com/Oblarg/whippyflight) Python tools.

## License

MIT or Apache-2.0, at your option.
