# Trainer & ModelState

Roxxel's `Trainer` is a curriculum-aware training orchestrator specifically designed for JAX and Flax NNX. 

It handles training loops, dynamic sequence and batch transitions, metric logging, Orbax checkpointing, and model evaluations with minimal boilerplate.

---

## Easiest Trainer Configuration

With Roxxel, you do not need to write custom training states, or explicitly instantiate checkpointers and loggers. Simply supply your model, optimizer, curriculum, and a `loss_fn`, along with directory paths for checkpoints and system logs.

```python
import jax
import optax
from flax import nnx
from roxxel import Roxxel, Phase, Curriculum, Trainer

# 1. Define Flax NNX model and optimizer
model = nnx.Linear(10, 5, rngs=nnx.Rngs(42))
tx = optax.sgd(0.01)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

# 2. Define the curriculum
phases = [Phase(steps=1000, batch_size=4, seq_len=10)]
curriculum = Curriculum(primary_streamer=Roxxel("./data_*.rox"), phases=phases)

# 3. Define the loss function
def loss_fn(model, batch):
    logits = model(batch[:, :-1].astype(jax.numpy.float32))
    targets = batch[:, 1:].astype(jax.numpy.float32)
    return jax.numpy.mean((logits - targets) ** 2)

# 4. Initialize the Trainer
# Setting paths for logger and checkpointer automatically initializes them
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    curriculum=curriculum,
    loss_fn=loss_fn,
    checkpointer="./checkpoints",
    logger="./run_logs",
    checkpoint_every=100,
    log_every=10
)

# 5. Run curriculum training
trainer.run()
```

---

## Core Features

### 1. Automated ModelState Creation
When you pass a standard JAX model and optimizer separately, the trainer constructs a `ModelState` object internally. It maintains:
- `state.model`: Reference to the Flax NNX Module.
- `state.optimizer`: Reference to the Flax NNX Optimizer.
- `state.step`: An `nnx.Variable` representing the global optimization step.

If you already have a pre-constructed custom state object containing `model` and `optimizer` attributes, the trainer automatically detects it for backward compatibility.

### 2. Internal JIT Train Step Compilation
The `Trainer` automatically defines and compiles a standard Flax JIT training step (`@nnx.jit`) on initialization. It executes:
- Forward pass through your `loss_fn`.
- Gradient computation via `nnx.value_and_grad`.
- Optimizer parameters update.
- Step counter incrementation.

### 3. Robust Loss wrapping
If your `loss_fn` returns multiple outputs (e.g. `(loss, aux_data)` or `{"loss": loss, "accuracy": acc}`), `Trainer` wraps it using `loss_wrapper` to ensure only the scalar loss is supplied to JAX gradient compilation, avoiding JAX compiler errors while preserving metrics.

### 4. Automatic Resource Management
If `logger` or `checkpointer` is passed as a directory path:
- The trainer initializes them internally.
- It executes training within the logger's asynchronous context manager to ensure system logs and metrics are flushed to disk even in the event of an OOM or hardware crash.
- It safely closes and flushes all Orbax checkpointers during final cleanup.

---

## API Reference

### Trainer
::: roxxel.trainer.Trainer
    options:
      show_root_heading: true
      heading_level: 3

### ModelState
::: roxxel.trainer.ModelState
    options:
      show_root_heading: true
      heading_level: 3
