# Roxxel

Welcome to the documentation for **Roxxel**: a zero-RAM, JAX-centric dataloading, streaming, and checkpointing & logging toolkit.

Roxxel is designed specifically for large-scale deep learning pipelines using JAX & Flax NNX. It provides OS-level memory mapping, zero-copy sharded device streaming, background model checkpointing, and thread-safe distributed logging.

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

## Navigation

* **[Tutorial / Cookbook](tutorial.md)**: A complete end-to-end JAX/Flax NNX training harness.
* **[Dataloader & Streaming](dataloader_and_streaming.md)**: Zero-RAM block loading and JAX-native multi-device data placement.
* **[Checkpointing](checkpointing.md)**: Non-blocking weight and optimizer state preservation using Orbax.
* **[Logging](logging.md)**: Zero-overhead logging with multi-host safety and automatic traceback capture.

