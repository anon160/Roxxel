# Roxxel 🚀 

**Zero-RAM, JAX-Centric Dataloading, Streaming, and Asynchronous Checkpointing & Logging Toolkit**

Roxxel is an ultra-lightweight, zero-bloat, high-performance toolkit designed specifically for large-scale JAX & Flax NNX deep learning training pipelines (such as State Space Models, Transformers, and SSMs like Xenron). 

By combining POSIX memory-mapped dataset sharding, high-performance async logging, and Flax NNX topology-agnostic asynchronous checkpointing, Roxxel provides a unified, framework-native pipeline that does away with heavy, over-engineered training frameworks.

---

## 🌟 The Four Pillars of Roxxel

### 1. Zero-RAM Sharded Block Dataloader (`roxxel.Roxxel`)
* **OS-Level Memory Mapping:** Maps multi-terabyte datasets directly into virtual memory via the operating system's kernel page cache using `numpy.memmap`. Consumes **exactly 0 bytes of Python RAM** for storage.
* **Dynamic Dtype Auto-Detection:** Automatically detects the data representation (e.g. `int32` token IDs, `float32` arrays, or `uint8` bytes) on compilation and stores it in a backward-compatible 32-byte footer (`ROXXEL02` format).
* **Precise O(1) Fast-Forwarding:** Instantly resumes streaming from any checkpointed step in under 1 millisecond using exact byte offsets—completely skipping the need to execute dummy fast-forward loops.

### 2. JAX-Native Streaming (`dataset.stream()`)
* **Zero Double-Copy Overhead:** Automatically chunks, shuffles, and places batches directly onto JAX device layouts (`jax.device_put`) using your Named Sharding Mesh. Avoids JAX default-device materialization bottlenecks and GPU/TPU OOM spikes.
* **Dynamic Step Calculation:** The stream returns a custom `RoxxelStream` object which natively exposes `len(stream)`—enabling you to instantly align learning rate schedules and progress bars.

### 3. Asynchronous Model Checkpointing (`roxxel.checkpoint.Checkpointer`)
* **Zero-Latency Async Storage:** Offloads state serialization to background threads using Orbax Checkpoint Manager, allowing your GPU/TPU accelerators to keep training without waiting for disk writes.
* **NNX Topology Agnostic:** Restores state PyTrees natively using abstract template evaluation, decoupling model architecture updates from saved weights.
* **Best-Loss Tracking:** Automatically monitors metric payloads and preserves the checkpoint achieving the lowest training loss (`best_mode='min'`).

### 4. Asynchronous JAX-Aware Logging (`roxxel.Logger`)
* **Zero-Overhead Async Execution:** Spawns a background thread queue (`QueueListener`) to process writes to standard output and disk files asynchronously. Zero interference with critical GPU/TPU execution.
* **Multi-Host TPU/GPU Pod Safety:** Automatically detects JAX rank and restricts logging to Rank 0, completely avoiding log corruption and process conflicts across multi-node pre-training clusters.
* **Atomic Exception Traceback Capture:** Implements robust context-manager (`with` statement) logic. If a TPU OOMs, crashes, or is forcefully interrupted, the queue instantly flushes to the log file and records the exact stack trace before bubbling the error up.

---

## 📦 Installation

Roxxel can be installed via `pip` directly from PyPI.

To install the core dataloader and async logging engine only:
```bash
pip install roxxel
```

To install the JAX-native asynchronous checkpointing extensions:
```bash
pip install roxxel[checkpoint]
```

---

## 🚀 End-to-End JAX/Flax NNX Training Cookbook

Here is a complete, real-world example showing how Roxxel integrates data compilation, sharded streaming, asynchronous system logging, JAX Named Sharding, and Orbax NNX checkpointing into a single, highly optimized training pipeline:

```python
import os
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils

from roxxel import Roxxel, Logger
from roxxel.checkpoint import Checkpointer

# --- 1. DATASET COMPILATION ---
# Let's compile tokenized integer sequences into 4KB uniform blocks
def token_generator():
    for i in range(1000):
        # yields numpy arrays of tokenized int32 IDs
        yield jnp.arange(128, dtype=jnp.int32)

rox = Roxxel("./wiki_*.rox")
rox.write(token_generator(), block_size=4096, max_shard_bytes=1024**3, separator=None)


# --- 2. HIGH-PERFORMANCE TRAINING HARNESS ENVIRONMENT ---
GLOBAL_SEED = 42
BATCH_SIZE = 32
SEQ_LEN = 1024
EPOCHS = 3
LR = 3e-4

# Open the dataset once to get the exact steps per epoch to define the scheduler
with Roxxel(filepath="./wiki_*.rox") as init_ds:
    steps_per_epoch = init_ds.estimate_steps(seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

total_train_steps = steps_per_epoch * EPOCHS

# Initialize your async text logger inside an atomic Context Manager.
# This guarantees that if a TPU crashes, OOMs, or is forcefully interrupted,
# the thread queue will instantly drain completely to 'run_delta/roxxel_system.log'
with Logger(log_dir="run_delta") as logger:
    logger.log_message("🚀 Initializing Distributed Pre-training Cluster...")

    # Initialize Flax NNX model tracking states using unified seed
    rngs = nnx.Rngs(GLOBAL_SEED)
    
    class SimpleSSM(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            self.embed = nnx.Embed(10000, 256, rngs=rngs)
            self.linear = nnx.Linear(256, 10000, rngs=rngs)
            
        def __call__(self, x):
            return self.linear(self.embed(x))
            
    model = SimpleSSM(rngs)
    
    # Setup Optax learning rate schedule
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=LR,
        warmup_steps=int(total_train_steps * 0.05),
        decay_steps=total_train_steps
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(scheduler))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # --- 3. ORBAX CHECKPOINT RESTORATION ---
    # Instantiate Checkpointer (saves parameters and optimizer natively)
    checkpointer = Checkpointer(checkpoint_path="./checkpoints", model=model, optimizer=optimizer)
    start_step = checkpointer.restore()
    logger.log_message(f"🔄 Checkpointer restored. Starting from step: {start_step}")

    # --- 4. SHARDED JAX STREAMING & TRAINING LOOP ---
    # Create distributed hardware sharding paths for Multi-Host TPU/GPU Pod scaling
    devices = jax.devices()
    mesh = Mesh(mesh_utils.create_device_mesh((len(devices),)), axis_names=('data',))
    data_sharding = NamedSharding(mesh, P('data', None))

    @nnx.jit
    def train_step(model, optimizer, batch):
        def loss_fn(model):
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
            return loss
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    for epoch in range(EPOCHS):
        logger.log_message(f"⏳ Starting Training Epoch {epoch + 1}/{EPOCHS}...")
        
        with Roxxel(filepath="./wiki_*.rox") as dataset:
            # Load hardware-sharded JAX device arrays instantly
            # (Epoch 0 resume fast-forwards instantly to start_step in O(1) time!)
            loader_stream = dataset.stream(
                seq_len=SEQ_LEN,
                batch_size=BATCH_SIZE,
                seed=GLOBAL_SEED,
                start_step=start_step if epoch == 0 else 0,
                mesh=mesh,
                data_sharding=data_sharding
            )
            
            # RoxxelStream supports len() natively for progress bars and scheduler checks!
            logger.log_message(f"Loaded {len(loader_stream)} steps remaining in this epoch.")
            
            for step_idx, batch in enumerate(loader_stream):
                loss = train_step(model, optimizer, batch)
                curr_step = start_step + step_idx if epoch == 0 else step_idx
                
                # Save asynchronously on a schedule (Orbax tracks the best model automatically!)
                if curr_step % 100 == 0:
                    logger.log_message(f"Step {curr_step} | Loss: {loss:.4f}")
                    logger.log_metrics_summary(step=curr_step, metrics={"loss": float(loss), "perplexity": float(jnp.exp(loss))})
                    checkpointer.save(curr_step, metrics_dict={"loss": loss})
                    
        start_step = 0  # Reset offset after completing epoch 0
```

---

## 🔄 API Evolution: The Old Way vs. The New Fused Way

Roxxel has been completely re-engineered to provide unified, non-blocking logs alongside zero-copy sharding, memory mapping, and background Orbax checkpointing for distributed deep learning.

| Feature | The Old Way (v0.1.0) | The New Fused Way (v0.5.x) |
| :--- | :--- | :--- |
| **Block Compilation** | Required wrapping writers in a separate external compiler class (`RoxxelBlockCompiler`) to manually group and pad inputs. | **100% Fused & Native**: The `rox.write()` API consumes arbitrary string/byte/numpy generators, automatically chunks them, handles padding, and writes to disk in one call. |
| **Data Types on Disk** | Restricted entirely to byte-level representations (`uint8`). Multi-byte dtypes (like tokenized `int32` IDs) were corrupted/split. | **Dynamic Dtype Metadata**: Automatically detects the datatype (e.g. `int32`, `float32`) on write, stores it in a 32B footer (`ROXXEL02`), and decodes perfectly on read. |
| **Shard Management** | Users had to write manual file rotation loops, file naming schemes, and offset tables to handle large datasets. | **Zero-Config Sharding**: Specify a glob path (e.g., `wiki_*.rox`) and `max_shard_bytes`. Roxxel handles shard rotation and virtualizes them into one contiguous list view. |
| **DL / JAX Streaming** | Required writing custom shuffling code, buffer management, and tedious boilerplate `jax.device_put` pipelines. | **Unified Causal Streaming**: The `dataset.stream()` API handles globally shuffled batching, O(1) step resumption, and automatic JAX sharded device placement with zero double-copy overhead. |
| **Model Checkpointing** | Standard training loops required manual `pickle`, custom JSON savers, or synchronous JAX disk blocks. | **Asynchronous Orbax (`Checkpointer`)**: Flax NNX model weights/optimizers are serialized concurrently in background threads with auto-computed best-loss tracking. |
| **Distributed System Logging** | Standard print statements caused GPU pipeline bottlenecks and multi-host log overlapping. | **Asynchronous Logger (`Logger`)**: Queue-based async writing offloaded to background threads with multi-host rank-zero filters and atomic exception traceback capturing. |

---

## ⚖️ License
MIT License. Feel free to use, modify, and distribute.
