//! The whippyflight plant fit from Betaflight blackbox logs.
//!
//! [`bbl`] decodes the logs, [`flight`] gates them, [`rows`] builds the
//! regressions that [`fit`] solves ([`motors`] by output error), and
//! [`diagnostics`] judges the result. [`scan`] indexes a log's flights,
//! [`noise`] extracts the sensor noise the sims replay, and [`occupancy`]
//! exports the gated motor speeds the occupancy plots draw.

pub mod airframe;
pub mod bbl;
pub mod diagnostics;
pub mod flight;
pub mod fit;
pub mod frame;
pub mod json;
pub mod motors;
pub mod noise;
pub mod num;
pub mod occupancy;
pub mod rows;
pub mod run;
pub mod scan;
pub mod units;
