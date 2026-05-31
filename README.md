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

### 1. Compiling raw data into uniform blocks
Pass an iterable stream of strings or raw bytes directly into the `write()` API. Roxxel will automatically group them into strictly uniform blocks (e.g., 4096-byte blocks) and write them to disk.

```python
from roxxel import Roxxel

# Generator yielding text documents of variable lengths
def text_generator():
    yield "The quick brown fox jumps over the lazy dog."
    yield "Generative AI and SSMs like Xenron are transforming sequences."
    yield "Roxxel delivers zero-RAM, highly efficient data loading."

# Instantiate and write (automatically shards at 1GB, blocks of 4KB)
rox = Roxxel("./wiki_*.rox")
rox.write(
    data_generator=text_generator(),
    block_size=4096,
    max_shard_bytes=1024**3,
    separator=b"\xff"
)
```

### 2. High-Performance Deterministic Streaming (NumPy & JAX)
Opening the dataset and streaming globally shuffled, JAX-sharded batches takes just a few lines. The `stream()` API handles circular prefetch buffering and PCIe hardware transfers asynchronously in the background:

```python
import jax
from roxxel import Roxxel

# 1. Open the virtualized multi-sharded dataset
with Roxxel("./wiki_*.rox") as dataset:
    # 2. Yield globally shuffled JAX-sharded device arrays automatically!
    # If JAX is installed, it outputs JAX arrays. Otherwise, standard NumPy arrays.
    dataloader = dataset.stream(
        seq_len=1024,
        batch_size=32,
        seed=42,
        start_step=0  # Supports instant O(1) checkpoint fast-forwarding!
    )
    
    for batch in dataloader:
        # batch is a JAX device-put array of shape (32, 1024) ready for TPU/GPU!
        outputs = train_step(state, batch)
```

---

## 🍳 Cookbooks

### A. Flat Token Reshaping (e.g. LLM Training)
If you want to bypass sequence boundaries entirely and treat the whole sharded database as a single, contiguous token stream:

```python
with Roxxel("./wiki_*.rox") as dataset:
    # Get a 1D memory-mapped view of the entire dataset across all shards
    # (Since len(dataset[0]) is uniform, this represents one solid contiguous sequence)
    flat_tokens = dataset.raw_data.view(np.uint16)
    
    # Reshape and train dynamically in NumPy
    total_sequences = len(flat_tokens) // 2048
    reshaped_dataset = flat_tokens[:total_sequences * 2048].reshape(total_sequences, 2048)
```

### B. Instant O(1) Checkpoint Resuming
To save training progress, simply checkpoint your current `step`. Resuming takes less than 1 millisecond as `Roxxel` instantly jumps past the consumed blocks using basic index arithmetic—completely skipping the need to execute dummy fast-forward loops:

```python
# During save:
step = current_step  # Save this integer in your checkpoint manager

# During restore:
with Roxxel("./wiki_*.rox") as dataset:
    # Instantly fast-forward and resume streaming from step 4500
    dataloader = dataset.stream(
        seq_len=1024,
        batch_size=32,
        seed=42,
        start_step=4500
    )
```

---

## ⚖️ License
MIT License. Feel free to use, modify, and distribute.
