//! Host-side simulation for [`whippyflight_control`]: the nonlinear plant
//! gen_wfg.py linearizes, and the stick flicks the evals and the
//! Configurator's WhippyFlight tab fly the controller through on it.

pub mod flick;
pub mod margin;
pub mod noise;
pub mod plant;
pub mod units;

pub use whippyflight_control::gains::static_blob;
