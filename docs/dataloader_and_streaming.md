# Dataloader & Streaming

Roxxel provides a zero-RAM, OS-level memory-mapped dataset manager designed for JAX/Flax pipelines. It virtualizes multiple dataset shards into a single contiguous stream and pipes batches directly onto JAX device sharding layouts.

---

## The Context Manager (`with` Statement)

Memory-mapping large datasets maps file segments into virtual memory. To prevent file descriptor leaks and ensure proper kernel resource cleanup, you should **always** use the Python context manager (`with` statement) when opening a Roxxel dataset:

```python
from roxxel import Roxxel

# Safely open, virtualize shards, and memory-map data
with Roxxel(filepath="/content/fineweb_edu_*.rox") as dataset:
    # Estimate total training steps
    steps = dataset.estimate_steps(seq_len=1024, batch_size=32)
    print(f"Dataset loaded. Total steps in epoch: {steps}")
    
    # Initialize JAX stream
    stream = dataset.stream(seq_len=1024, batch_size=32, seed=42)
    for batch in stream:
        # Train model here
        pass

# The files are automatically closed and memory-unmapped clean here!
```

### Why is this necessary?
1. **POSIX Page Cache:** When the context manager exits, Roxxel automatically closes all open file descriptors and cleans up mapping tables.
2. **Resource Safety:** If your training loop raises an exception, the context manager guarantees the memory is unmapped immediately, avoiding memory corruption or locked file handlers on your cloud virtual machine.

---

## Compiling Datasets with Tokenizers

When compiling raw datasets into Roxxel's uniform block format, you can integrate a Hugging Face tokenizer (from either the `tokenizers` or `transformers` library) directly. 

Providing a `tokenizer` to `Roxxel.write()` enables:
* **On-the-Fly Tokenization**: Strings yielded by the generator are automatically tokenized.
* **Native EOS Document Separator**: The tokenizer's End-Of-Sequence (EOS) token ID is automatically extracted and injected as a separator between documents.
* **Auto-Dtype Resolution**: If the tokenizer's vocabulary size is greater than 65,535, Roxxel compiles the dataset with `int32` token IDs; otherwise, it uses `int16` for optimal storage.
* **Padding Alignment**: The block padding uses the tokenizer's `pad_token_id` (falling back to `eos_token_id`).

### Example: Compiling Text Data with a Tokenizer

```python
from roxxel import Roxxel
from tokenizers import Tokenizer

# 1. Load a Hugging Face tokenizer (or pass a model ID string like "meta-llama/Llama-3-8B")
tokenizer = Tokenizer.from_pretrained("meta-llama/Llama-3-8B")

# 2. Define text stream generator
def text_generator():
    yield "The quick brown fox jumps over the lazy dog."
    yield "JAX-native streaming is extremely fast."

# 3. Compile raw text into uniform 4KB block archives
rox = Roxxel("./data/dataset_*.rox")
rox.write(
    text_generator(),
    tokenizer=tokenizer,
    block_size=4096,
    max_shard_bytes=1024**3
)
```

For backward compatibility, you can write raw pre-tokenized binary files by omitting `tokenizer` and specifying a binary `separator` (e.g. `separator=b"\x00"` or `separator=None`).

---

## JAX-Native Device Sharding

Roxxel streams JAX arrays directly into your hardware topology (TPU Mesh or GPU grid) without copying data twice or causing CPU-to-GPU materialization spikes.

Here is how to stream batches directly into a JAX Named Sharding mesh:

```python
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from roxxel import Roxxel

# 1. Setup multi-device JAX mesh
devices = jax.devices()
mesh = Mesh(mesh_utils.create_device_mesh((len(devices),)), axis_names=('data',))
data_sharding = NamedSharding(mesh, P('data', None))

# 2. Stream sharded batches
with Roxxel(filepath="./data/fineweb_edu_*.rox") as dataset:
    stream = dataset.stream(
        seq_len=1024,
        batch_size=32,
        seed=42,
        mesh=mesh,
        data_sharding=data_sharding
    )
    
    for batch in stream:
        # 'batch' is already placed on JAX devices matching 'data_sharding'!
        assert isinstance(batch, jax.Array)
        assert batch.sharding == data_sharding
```

---

## API Reference

::: roxxel.core.Roxxel
    options:
      show_root_heading: true
      heading_level: 3
