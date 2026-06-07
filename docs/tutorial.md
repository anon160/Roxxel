# End-to-End Tutorial

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
