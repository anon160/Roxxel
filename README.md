# Roxxel 🚀 

**Zero-RAM JAX Dataloader, Asynchronous Checkpointer & Logging Trainer for Flax NNX**

Roxxel is a zero-RAM, ultra-lightweight, and exceptionally fast dataset manager and pre-training orchestrator designed specifically for distributed JAX/Flax NNX clusters. 

By utilizing virtualized POSIX memory-mapped dataset sharding, background asynchronous Orbax checkpointing, and thread-safe distributed logging, Roxxel completely does away with heavy, complex, and over-engineered training frameworks.

---

## 🌟 Key Features

- **OS-Level Memory Mapping (`mmap`)**: Maps multi-terabyte datasets directly into virtual memory via the operating system's kernel page cache. Consumes exactly **0 bytes of Python RAM** for dataset storage.
- **Unified Causal Streaming**: Automatically chunks, shuffles, and loads batches directly onto JAX device layouts (`jax.device_put`) using your Named Sharding mesh. Exposes the exact step count (`len(stream)`).
- **Instant Offset Seeking**: Resumes streaming from any step index in under 1 millisecond using binary offsets—completely skipping the need to execute slow dummy fast-forward loops.
- **Dynamic Dtype Auto-Detection**: Detects datatypes (e.g. `int32` token IDs, `float32` arrays, or raw text bytes) on compilation, writes them in a backward-compatible format, and decodes them perfectly on read.
- **Multi-Dataset Blending / Mixing**: Blends a primary dataset with multiple secondary datasets using weight ratios. Automatically handles dataset exhaustion mid-phase by re-normalizing weights on the fly.
- **Curriculum Schedule Timelines**: Supports dynamic sequence length extension (e.g., shifting from 1K to 32K context windows) and batch size changes at precise training step boundaries.
- **Topology-Agnostic Checkpointing**: Offloads PyTree serialization asynchronously to background threads via Orbax Checkpoint Manager. Restores parameters natively using abstract templates, meaning model definition changes do not break older saved weights.
- **Multi-Host TPU/GPU Pod Safety**: Restrictions ensure only Rank 0 writes stdout prints, system logs, and metrics CSV files, avoiding multi-process locking contention and terminal clutter.

---

## 🤖 What Roxxel Handles Automatically (So You Don't Have To)

Roxxel was engineered to remove the friction of writing accelerator-optimized JAX code. It handles the following tasks automatically under the hood:

1. **JIT Train Step Compilation**: On initialization, `Trainer` automatically compiles your JIT training step (`@nnx.jit`). You don't need to write or trace JAX functions manually.
2. **Loss Unpacking & Wrapping**: If your loss function returns a tuple/list (like `(loss, aux_data)`) or a dictionary (like `{"loss": loss, "perplexity": ppl}`), Roxxel automatically extracts the scalar loss for gradients (`nnx.value_and_grad`), avoiding compiler crashes.
3. **Auto-Initialized ModelState**: You don't have to write state wrapper boilerplate (e.g. `TrainState` classes). The trainer dynamically bundles your model, optimizer, and training step counter.
4. **Auto-Initialized Checkpointer & Logger**: If you pass a folder path to the `save_path` parameter, Roxxel automatically initializes the asynchronous Orbax Checkpointer and multi-threaded Logger under that directory, saving checkpoints in `save_path/checkpoints` and logs directly in `save_path`.
5. **Dynamic Stream Re-instantiation**: During curriculum phase transitions (e.g. when sequence length changes), Roxxel automatically closes the active stream, computes completed offsets, and swaps the dataset streams instantly.
6. **Exhaustion Re-normalization**: If one of your mixed datasets runs out of records mid-training, Roxxel removes it from the choice pool and re-normalizes the weights of the remaining active datasets to prevent training stalls.
7. **Crash-Safe Log Flushing**: In the event of an OOM, crash, or exception, the logger intercepts the exception, logs the stack trace to the system log, and flushes all pending file and stdout writes before bubbling the error.

---

## 📦 Installation

To install the core dataloader and async logging engine only (no JAX required):
```bash
pip install roxxel
```

To install the JAX-native asynchronous trainer and checkpointing extensions:
```bash
pip install roxxel[checkpoint]
```

---

## 🚀 Quick Start Cookbook

Here is a complete, zero-boilerplate example showing how to initialize a model, define a curriculum schedule, and execute pre-training using the consolidated `Trainer`:

```python
import jax
import optax
from flax import nnx
from roxxel import Roxxel, Phase, Curriculum, Trainer

# 1. Initialize Flax NNX model and optimizer
model = nnx.Linear(10, 5, rngs=nnx.Rngs(42))
tx = optax.sgd(0.01)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

# 2. Define the curriculum (e.g., Phase 1: 1000 steps, Phase 2: 500 steps)
phases = [
    Phase(steps=1000, batch_size=16, seq_len=128),
    Phase(steps=500, batch_size=4, seq_len=512)
]
curriculum = Curriculum(
    primary_streamer=Roxxel("./wiki_tokens_*.rox"), 
    phases=phases
)

# 3. Define the loss function
def loss_fn(model, batch):
    logits = model(batch[:, :-1].astype(jax.numpy.float32))
    targets = batch[:, 1:].astype(jax.numpy.float32)
    return jax.numpy.mean((logits - targets) ** 2)

# 4. Initialize the Trainer
# Setting save_path automatically initializes Checkpointer, Logger, and ModelState
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

## 📖 Learn More

For complete documentation, design guides, and API specs, visit the [Roxxel Documentation Site](https://anon160.github.io/Roxxel/):
- **[Why Roxxel?](https://anon160.github.io/Roxxel/why_roxxel/)**: Design philosophy and comparisons to PyTorch/TensorFlow dataloaders.
- **[End-to-End Tutorial](https://anon160.github.io/Roxxel/tutorial/)**: Full cookbook with hardware sharding and dataset blending.
- **[Dataloader & Streaming](https://anon160.github.io/Roxxel/dataloader_and_streaming/)**: Deep dive into block virtualization and streams.
- **[Curriculum Blending](https://anon160.github.io/Roxxel/curriculum/)**: How to mix datasets and define phases.
- **[Trainer Orchestration](https://anon160.github.io/Roxxel/trainer/)**: Full Trainer and ModelState specifications.
- **[Asynchronous Checkpointing](https://anon160.github.io/Roxxel/checkpointing/)**: Orbax asynchronous serialization details.
- **[Asynchronous Logging](https://anon160.github.io/Roxxel/logging/)**: Rank-zero queue writing details.

---

## ⚖️ License
MIT License. Feel free to use, modify, and distribute.
