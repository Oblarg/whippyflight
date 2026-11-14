//! WhippyFlight — a gain-scheduled LQR attitude-rate controller for
//! multirotors.
//!
//! The control law uses precomputed gain matrices, generated offline by
//! solving the discrete-time algebraic Riccati equation at a grid of
//! operating points, and loaded from a gain blob ([`gains`]).
//!
//! # Architecture
//!
//! This crate is a pure control law: it takes sensor data in and produces
//! motor duty out, and owns no hardware, globals or allocation.  A firmware
//! adapter (e.g. the Betaflight plugin) is the HAL: it reads sensors,
//! drives motors, logs telemetry, stores the gain blob, and converts its
//! own units and conventions into this crate's ([`types`]).
//!
//! The per-tick entry point is [`controller::ControllerState::update`]
//! (or `update_with_target`, fed by [`setpoint::SetpointEstimator`]).

#![cfg_attr(not(feature = "std"), no_std)]

pub mod allocation;
pub mod controller;
pub mod gains;
pub mod interpolation;
pub mod setpoint;
pub mod types;
