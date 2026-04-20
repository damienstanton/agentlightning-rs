# agentlightning

Rust port of Microsoft's [Agent Lightning](https://github.com/microsoft/agent-lightning) reinforcement learning framework.

## Overview

Agent Lightning provides environment-independent RL training through structured span recording. This Rust implementation eliminates Python fragility and enables single-binary deployment.

## Features

- 🎯 **Span Collection**: Structured recording of agent interactions (Observations, Actions, Rewards)
- 💾 **Lightning Store**: Embedded Sled-based persistent storage with automatic indexing
- 🧠 **Algorithm Interface**: Trait-based RL algorithm abstraction
- 🔄 **Async Trainer**: Configurable training loop with batching and metrics

## Quick Start

```rust
use agentlightning::{
    LightningStore, ObservationSpan, RewardSpan, Span,
    algorithm::RewardAggregator, Trainer, TrainerConfig,
};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize store
    let store = Arc::new(LightningStore::open("~/.agentlightning")?);
    
    // Emit spans
    let obs = Span::Observation(
        ObservationSpan::new(json!({"step": 1}))
            .with_task("demo")
            .with_agent("agent-1")
    );
    store.insert_span(&obs)?;
    
    // Train
    let config = TrainerConfig {
        task_id: Some("demo".into()),
        batch_size: 50,
        ..Default::default()
    };
    
    let mut trainer = Trainer::new(store.clone(), config);
    let mut algo = RewardAggregator::default();
    let results = trainer.run(&mut algo).await?;
    
    Ok(())
}
```

## Architecture

```
agentlightning/
├── span.rs        # Span types (Observation, Action, Reward)
├── collector.rs   # SpanCollector trait + implementations
├── store.rs       # Sled-based persistent storage
├── algorithm.rs   # LightningAlgorithm trait
├── trainer.rs     # Training loop orchestration
```

## Features

- **Default**: Core Lightning functionality with composable crate boundaries.

```toml
[dependencies]
agentlightning = { path = "crates/agentlightning" }
```

## Testing

```bash
# Run all tests across the workspace
cargo test --workspace

# Run specific module tests in the core crate
cargo test -p agentlightning-core span::tests
```

The test suite covers span types, storage, algorithms, and training across the workspace crates.

## References

- [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning)
- [Research Paper](https://arxiv.org/abs/2508.03680)
