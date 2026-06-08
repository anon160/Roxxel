# Why Roxxel?

Large-scale model pre-training on modern accelerators (TPUs and GPUs) is extremely fast. However, feed-forward training loops are frequently bottle-necked by **data loading, preparation, and auxiliary operations** (such as checkpointing and system logging).

Standard tools in the Python ecosystem (like PyTorch `DataLoader` or TensorFlow `tf.data`) were designed for PyTorch/TF execution models and often introduce severe overhead, complex subprocess communication, and GPU/TPU memory issues. 

**Roxxel** was built from the ground up to solve these specific bottlenecks for JAX & Flax NNX training pipelines.

---

## The Core Philosophy: OS-Level Memory Mapping

Standard dataloaders spawn multiple Python background processes (`num_workers > 0`), which load records, tokenize/process them in parallel, and transfer them back to the main training process via POSIX shared memory or IPC pipes. This design creates several problems:
1. **Shared Memory OOMs**: Large batch sizes or long sequence lengths frequently exhaust shared memory (`/dev/shm`), crashing the pre-training run.
2. **CPU-to-GPU Bottlenecks**: Moving memory between multiple Python processes and copying it to host memory, then copying it to GPU/TPU device memory, creates a *double-copy overhead*.
3. **Stalls at Boundaries**: Spawning and joining processes at phase boundaries or epoch transitions causes training to stall.

### How Roxxel is Different
Roxxel decouples data loading from Python's interpreter. Instead of spawning worker subprocesses, Roxxel virtualizes your sharded dataset files using **POSIX Memory Mapping (`mmap`)**. 

When your training loop requests a batch, the operating system kernel's page cache resolves the byte offsets of the records in memory instantly (`O(1)` time) with **0ms latency** and **exactly 0 bytes of Python RAM overhead**.

```
Standard DL Dataloader:
[Disk] -> [Worker Processes] -> [IPC Pipes/Shared Memory] -> [Main Process] -> [Host RAM] -> [Device VRAM] (Double-Copy!)

Roxxel Dataloader:
[Disk] -> [OS Page Cache / mmap] -> [JAX Device Array / Sharded Mesh] (Single Zero-Copy Direct Placement!)
```

---

## Comparison: PyTorch vs. TensorFlow vs. Roxxel

| Feature | PyTorch `DataLoader` | TensorFlow `tf.data` | Roxxel |
| :--- | :--- | :--- | :--- |
| **Primary Framework** | PyTorch | TensorFlow | **JAX / Flax NNX** |
| **Memory Management** | Host RAM Copy / Multiprocessing | C++ Runtime Graph | **Zero-RAM Virtual memory-mapped (`mmap`)** |
| **Resume Seek Latency** | Slow (Must iterate through dummy steps) | Slow (Graph state restore) | **Instant under 1ms (`O(1)` offset seek)** |
| **Double-Copy Overhead** | High (IPC pipes -> host -> device) | High (Host memory -> device copy) | **Zero (Direct placement on sharded JAX device mesh)** |
| **Multi-Host TPU Pods** | Requires manual rank filtering | Complex coordinator graphs | **Automatically detects process index (Rank 0 restriction)** |
| **Curriculum Aware** | Requires custom wrapper libraries | Requires complex dynamic shapes | **First-class native curriculum phase transitions** |

---

## What Roxxel Handles Automatically (So You Don't Have To)

Roxxel is not just a dataloader; it is an out-of-the-box pre-training loop orchestrator. Here is what it does automatically:

1. **JIT Train Step Compilation**: Automatically compiles your trainer's `train_step` using `@nnx.jit` on initialization, meaning you don't have to deal with manual JAX function tracing.
2. **Loss Unpacking & Wrapping**: If your loss function returns multiple outputs (e.g. `(loss, auxiliary_metrics)` or `{"loss": loss, "accuracy": acc}`), Roxxel automatically extracts the scalar loss for gradients, avoiding JAX compiler errors while preserving metrics.
3. **Dataset Exhaustion Re-normalization**: During multi-dataset blending, if any secondary dataset is fully consumed, the stream automatically drops it from the choice pool and re-normalizes the weights of the remaining active datasets to prevent crashes.
4. **Asynchronous Checkpointing**: Offloads checkpoints asynchronously to background threads via Orbax on demand, ensuring your TPU/GPU training never blocks for disk writes.
5. **Crash-Safe Queue Logging**: Runs log writing asynchronously on a background thread. If the training loop crashes or OOMs, the queue is caught by context managers, automatically logs the uncaught stack trace to files, and flushes everything cleanly to disk before bubbling the exception.
