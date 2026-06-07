# Checkpointing

Writing multi-gigabyte neural network weights and optimizer states to disk can freeze your GPU/TPU accelerators for several seconds, reducing hardware utilization. 

Roxxel's `Checkpointer` leverages **Orbax Checkpoint Manager** to offload PyTree serialization to background threads, allowing your training loop to continue JAX/Flax calculations immediately.

---

## Model & Optimizer Save/Restore Flow

Here is a complete example showing how to initialize a model, configure an Optax optimizer using Flax NNX, and manage state restoration and periodic saving:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from roxxel.checkpoint import Checkpointer

# 1. Initialize Flax NNX model and optimizer
class SimpleModel(nnx.Module):
    def __init__(self, rngs):
        self.linear = nnx.Linear(10, 5, rngs=rngs)
    def __call__(self, x):
        return self.linear(x)

rngs = nnx.Rngs(42)
model = SimpleModel(rngs)
tx = optax.adam(1e-3)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

# 2. Instantiate the Checkpointer
checkpointer = Checkpointer(
    checkpoint_path="./checkpoints",
    model=model,
    optimizer=optimizer,
    max_to_keep=3
)

# 3. Restore existing checkpoint if present
start_step = checkpointer.restore()
print(f"Resumed from step: {start_step}")

# 4. Training Loop
for step in range(start_step, 1000):
    # Train step...
    loss_val = 0.52
    
    # Save asynchronously (does not block JAX compilation or execution)
    if step % 100 == 0:
        checkpointer.save(
            step=step,
            metrics_dict={"loss": loss_val}
        )
```

---

## Core Features

### 1. Zero-Latency Background Offloading
When you call `checkpointer.save(...)`, Roxxel splits the Flax NNX states, packages them into abstract Orbax structures, and transfers the I/O execution to a background thread pool. Your TPU/GPU training is not blocked.

### 2. NNX Topology Agnostic
Roxxel's checkpointer avoids saving model architecture graphs directly to disk. Instead, it queries an abstract template of your model on restoration. This decoupling means you can update your Python model definition (e.g. adding layers or modifying attributes) without breaking compatibility with older saved weights.

### 3. Automated Best-Loss Tracking
Under the hood, the checkpointer tracks the evaluation metrics you pass via `metrics_dict`. By default, it automatically identifies and preserves the checkpoint achieving the lowest training loss (`best_mode='min'`), ensuring you never lose your best model state.

---

## API Reference

::: roxxel.checkpoint.Checkpointer
    options:
      show_root_heading: true
