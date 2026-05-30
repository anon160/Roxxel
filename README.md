# Roxxel 🚀

**Zero-RAM, Multi-Modal, Sharded Binary Dataset Manager**

Roxxel is an ultra-lightweight (~300 lines of plain Python), zero-dependency (except NumPy) binary dataset format and reader designed for high-performance deep learning pipelines. 

By implementing the standard Python sequence protocol over native `numpy.memmap` views, Roxxel virtualizes massive, multi-sharded, variable-length datasets on-disk as a simple, continuous in-memory list.

---

## 💡 Motivation

Mainstream deep learning data loaders—such as **PyTorch's `DataLoader`**, **Google's `Grain`**, and **TensorFlow's `tf.data`**—attempt to handle every aspect of the data pipeline (I/O, caching, multiprocessing, shuffling, collation, and transformations) in a single, massive monolithic system. This inevitably leads to severe operational friction:

* **PyTorch DataLoader**: Relying on multiple workers (`num_workers > 0`) spawns child processes that trigger Python's `fork` mechanism. This frequently results in massive memory leaks due to copy-on-write page sharing bugs in Python's GIL. Furthermore, debugging opaque subprocess deadlocks and socket/IPC exhaustion is incredibly frustrating.
* **Google Grain**: While powerful, it introduces a heavyweight dependency footprint and complex pipeline building abstractions that are difficult to customize or run outside of JAX-specific training pipelines.
* **TensorFlow tf.data**: Building robust tf.data pipelines is highly complex. Additionally, it forces you to use the opaque `TFRecord` binary format, which cannot be easily inspected or read without pulling in the massive, multi-gigabyte TensorFlow library as a dependency.

### The Roxxel Philosophy
Roxxel shifts the architectural boundary by practicing the **Unix philosophy of doing one thing and doing it well**. It handles only the hardest, most critical parts of storage—**safe contiguous file packing, zero-RAM memory mapping, and O(1) seek indexing**—and leaves all batching, threading, and transformations to plain, standard Python and NumPy code.

---

## 🌟 Unique Benefits of Roxxel

1. **Zero-RAM Overhead**: Roxxel maps your dataset directly into virtual memory via the operating system's kernel page cache using `numpy.memmap`. Even for multi-terabyte datasets, it consumes **exactly 0 bytes of Python RAM** for the data.
2. **100% Framework Agnostic**: Because Roxxel is built purely on Python standard libraries and NumPy, it is entirely decoupled from any ML framework. You can use the exact same Roxxel dataset across **PyTorch, JAX, TensorFlow, or pure CPU environments** with zero code changes.
3. **No Multiprocessing Deadlocks**: Because reading from memory maps is natively thread-safe and extremely fast, you can implement high-performance, asynchronous loading using simple Python threads (`threading.Thread`) or thread pools. You never have to worry about subprocess IPC bottlenecks or fork-related deadlocks.
4. **Modality-Agnostic Variable-Length Records**: Unlike rigid formats, Roxxel accepts arbitrary variable-length binary payloads contiguously. You can store JPEGs, MP4 clips, text token arrays, or audio samples in a single, unified structure with zero padding waste.
5. **Clean Sharded Portability**: Roxxel automatically splits massive datasets into sequentially numbered shards during writes. During reads, it seamlessly virtualizes them into a single continuous sequence using fast binary search boundaries. Shards are easy to distribute, copy, and stream over networks.

---

## 🛠️ File Format Architecture

To prevent **header contamination** (where inline metadata blocks corrupt flat memory maps), Roxxel writes your entire dataset into a single contiguous binary file with a trailing index table:

```
+-------------------------------------------------------------+
|                                                             |
| 1. RAW CONTIGUOUS PAYLOAD DATA SECTION                      |
|    (No headers, no prefixes, completely clean bytes)        |
|                                                             |
+-------------------------------------------------------------+
|                                                             |
| 2. TRAILING INDEX TABLE SECTION                             |
|    (Flat array of uint64 offsets pointing to record ends)   |
|                                                             |
+-------------------------------------------------------------+
| 3. FOOTER (Exactly 24 bytes)                                |
|    [Total Records (8B)] [Raw Data Size (8B)] [MAGIC (8B)]  |
+-------------------------------------------------------------+
```

Because the raw data section is completely uninterrupted, you can interpret the entire archive as a single contiguous array in one line (e.g. for LLM token pre-training) or resolve individual records in $O(1)$ constant time.

## 📦 Installation

Roxxel can be installed via `pip` directly from PyPI:

```bash
pip install roxxel
```

---

## 🚀 Getting Started

Simply copy `roxxel.py` into your project.

### 1. Writing a Single-File Dataset
```python
from roxxel import Roxxel

# Define a generator that yields raw byte payloads
def byte_stream():
    for i in range(100):
        yield bytes([i] * 50)  # Yield raw bytes

rox = Roxxel("./dataset.rox")
rox.write(byte_stream())
```

### 2. Writing a Sharded Dataset
Specify `max_shard_bytes` to automatically split massive data streams into dynamically capped shards (e.g., `dataset_0000.rox`, `dataset_0001.rox`):
```python
# Limit each shard to 2GB
rox.write(byte_stream(), max_shard_bytes=2 * 1024 * 1024 * 1024)
```

### 3. Reading and Shuffling (Sequence API)
Roxxel supports glob patterns and Python lists. It virtualizes all matching shards into a single read-only sequence supporting index lookups, negative indices, and slicing:
```python
import numpy as np
from roxxel import Roxxel

# Read and virtualize all shards matching the glob pattern
with Roxxel("./dataset_*.rox") as dataset:
    print("Total virtual records:", len(dataset))
    
    # 1. O(1) single index lookup
    record = dataset[42]
    
    # 2. Slice lookup
    subset = dataset[10:20]
    
    # 3. Global Shuffling (handled in three lines of plain NumPy!)
    shuffled_indices = np.random.permutation(len(dataset))
    for idx in shuffled_indices:
        shuffled_record = dataset[idx]  # seek & load happens instantly in page cache
```

---

## 🍳 Cookbooks

### A. Flat Token Streaming (e.g., LLM Training)
If you are doing LLM pre-training, you want to treat your entire dataset as one continuous stream of tokens. Roxxel allows you to ignore record boundaries and read the raw contiguous mapped memory directly:

```python
with Roxxel("./tokens.rox") as dataset:
    # Cast the entire mapped raw bytes section directly into uint16 tokens
    tokens = dataset.raw_data.view(np.uint16)
    
    # Chunk and batch locally in NumPy:
    seq_len = 2048
    total_sequences = len(tokens) // seq_len
    reshaped_batches = tokens[:total_sequences * seq_len].reshape(total_sequences, seq_len)
```

### B. High-Performance Asynchronous Prefetch Dataloader
If you are training on high-performance GPUs, you want to pre-load batches on a background CPU thread to completely prevent GPU starvation:

```python
import queue
import threading
import numpy as np
from roxxel import Roxxel

def async_dataloader(rox_pattern, batch_size=32, prefetch_batches=4, seed=42):
    dataset = Roxxel(rox_pattern)
    dataset.open()
    
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    q = queue.Queue(maxsize=prefetch_batches)
    
    def producer():
        for start_idx in range(0, len(indices), batch_size):
            batch_picks = indices[start_idx : start_idx + batch_size]
            
            # Fetch and decode/stack
            batch_data = [dataset[idx] for idx in batch_picks]
            q.put(batch_data)
        q.put(None)  # EOF
        dataset.close()

    # Start I/O in the background
    threading.Thread(target=producer, daemon=True).start()
    
    # Yield batches to the training loop
    while True:
        batch = q.get()
        if batch is None:
            break
        yield batch
```

---

## ⚖️ License
MIT License. Feel free to use, modify, and distribute.
