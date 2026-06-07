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

For a complete, real-world cookbook showing how Roxxel integrates data compilation, sharded streaming, asynchronous system logging, JAX Named Sharding, and Orbax NNX checkpointing into a single training pipeline, please refer to the [Roxxel Tutorial Documentation](https://anon160.github.io/Roxxel/tutorial/).

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
