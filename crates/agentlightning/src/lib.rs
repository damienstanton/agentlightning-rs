//! Agent Lightning - Reinforcement Learning Platform
//!
//! This crate serves as the facade for the Modular Brain Architecture.
//! It re-exports core components and provides a Runtime Factory for
//! hot-swappable algorithms.

// Re-export core components
pub use agentlightning_core::*;

pub mod factory;
pub mod harness;

// Re-export new runtime components
pub use factory::{AlgorithmConfig, BrainFactory};
pub use harness::TrainingHarness;
