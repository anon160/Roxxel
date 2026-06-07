# Roxxel 🚀

Welcome to the documentation for **Roxxel**: a zero-RAM, JAX-centric dataloading, streaming, and asynchronous checkpointing & logging toolkit.

Roxxel is designed specifically for large-scale deep learning pipelines using JAX & Flax NNX. It provides OS-level memory mapping, zero-copy sharded device streaming, background model checkpointing, and thread-safe distributed logging.

---

## Navigation

* **[Dataloader & Streaming](dataloader_and_streaming.md)**: Zero-RAM block loading and JAX-native multi-device data placement.
* **[Asynchronous Checkpointing](checkpointing.md)**: Non-blocking weight and optimizer state preservation using Orbax.
* **[Asynchronous Logging](logging.md)**: Zero-overhead logging with multi-host safety and automatic traceback capture.

For installation instructions and complete pre-training cookbooks, see the [Roxxel README](https://github.com/anon160/Roxxel).
