# WhippyFlight beta

WhippyFlight replaces Betaflight's PIDs with a gain-scheduled LQR controller fitted to your own quad from a blackbox log. You need the Configurator to fit and upload gains, and the firmware to fly them.

| Repository | Branch | What |
|------------|--------|------|
| [Oblarg/betaflight-configurator](https://github.com/Oblarg/betaflight-configurator) | `whippyflight` | Configurator with the WhippyFlight tab |
| [Oblarg/betaflight](https://github.com/Oblarg/betaflight) | `whippyflight` | Betaflight firmware with the WhippyFlight controller |
| [Oblarg/whippyflight](https://github.com/Oblarg/whippyflight) | `main` | The controller and fitting tools ([PyPI](https://pypi.org/project/whippyflight-betaflight/), [crates.io](https://crates.io/crates/whippyflight-control)) |

## Prerequisites

- [Rust](https://rustup.rs)
- [Node.js 24](https://nodejs.org) (or `nvm install 24`)
- [Python 3.11 or newer](https://www.python.org/downloads/) (macOS: python.org or Homebrew; Apple's `/usr/bin/python3` is too old)
- Tauri's [system dependencies](https://v2.tauri.app/start/prerequisites/) (Xcode command-line tools on macOS, Visual Studio C++ build tools on Windows)

## Configurator

```sh
git clone -b whippyflight https://github.com/Oblarg/betaflight-configurator.git
cd betaflight-configurator
npm ci
npm run tauri:dev     # or `npm run tauri:build` for an installable app
```

On the WhippyFlight tab, press **Set up** next to Python once. It makes a Python environment in the app's data folder and installs the fitting tools from PyPI.

## Firmware

```sh
git clone -b whippyflight https://github.com/Oblarg/betaflight.git
cd betaflight
make arm_sdk_install
rustup target add thumbv7em-none-eabihf     # F4, F7, H7; thumbv8m.main-none-eabihf for H5
make CONFIG=<your board> OPTIONS=USE_WHIPPYFLIGHT
```

Flash the `.hex` from `obj/` with the Configurator's Firmware Flasher (**Load Firmware [Local]**), then restore your settings.

## Fitting and flying

1. Enable bidirectional DShot and check the motor pole count; gains are scheduled on measured motor speed.
2. Log a flight at a high blackbox rate with motor RPM recorded, flying hard on every axis. Ensure USE_WHIPPYFLIGHT is disabled in the CLI; the quad will not arm with whippyflight enabled but no gain blob loaded, and the log analyzer that generates the fit does not rely on any whippyflight-specific data.
3. On the WhippyFlight tab, choose a working directory, **Import logs…** and **Import CLI dump…** (`dump` in the CLI), then **Fit** and **Compute gains**.
4. Fix anything under "Before WhippyFlight will arm" (acro only: no self-levelling modes, `failsafe_procedure = DROP`), then **Upload gains**.
5. Enable USE_WHIPPYFLIGHT in the CLI.

Report problems with the log, the working folder's `fits/` and `gains/`, and a screenshot of the tab.
