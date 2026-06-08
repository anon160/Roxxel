# Trainer & ModelState

Roxxel's `Trainer` is a curriculum-aware training orchestrator specifically designed for JAX and Flax NNX. 

It handles training loops, dynamic sequence and batch transitions, metric logging, Orbax checkpointing, and model evaluations with minimal boilerplate.

---

## Easiest Trainer Configuration

With Roxxel, you do not need to write custom training states, or explicitly instantiate checkpointers and loggers. Simply supply your model, optimizer, curriculum, and a `loss_fn`, along with a unified `save_path` directory.

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
# Setting save_path automatically initializes the checkpointer and logger
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    curriculum=curriculum,
    loss_fn=loss_fn,
    save_path="./run_delta",
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
If `save_path` is passed, the trainer automatically initializes:
- A `Checkpointer` located in `save_path/checkpoints`.
- A `Logger` saving metrics and system logs directly inside `save_path`.

Alternatively, you can pass custom checkpointer and logger instances or individual overrides as paths directly to `checkpointer` and `logger` arguments.

The trainer automatically executes all process-critical training steps within the logger's asynchronous context manager to guarantee tracebacks are logged and flushing occurs even during training crashes. It also executes asynchronous checkpointer flushes and close routines in final cleanup hooks.

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
