import os
import jax
from flax import nnx
import orbax.checkpoint as ocp

class CheckpointHandler:
    def __init__(self, checkpoint_path: str, model: nnx.Module, optimizer: nnx.Optimizer, max_to_keep: int = 3):
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.model = model
        self.optimizer = optimizer
        self.graphdef, _ = nnx.split((self.model, self.optimizer))
        async_opts = ocp.options.AsyncOptions(
            timeout_secs=1200,                # Give background threads 20 minutes to stream out to storage
            create_directories_asynchronously=True
        )

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep, 
            create=True,
            # Pass the configured async block into its designated slot instead of raw keywords
            async_options=async_opts          
        )
        self.mngr = ocp.CheckpointManager(self.checkpoint_path, options=options)

    def save(self, step: int, val_loss: float):
        """Extracts the global variable states and saves them directly as a raw PyTree."""
        _, state = nnx.split((self.model, self.optimizer))
        self.mngr.save(
            int(step),
            args=ocp.args.StandardSave(state),
            metrics={'val_loss': float(val_loss)}
        )
        if jax.process_index() == 0:
            print(f"💾 Asynchronous state PyTree saved securely at step {step}")

    def restore(self) -> int:
        """Restores parameters and optimizer tracking vectors natively without dictionary nesting."""
        latest_step = self.mngr.latest_step()
        if latest_step is None:
            if jax.process_index() == 0:
                print("🆕 No cluster checkpoint found. Starting fresh run from scratch.")
            return 0

        if jax.process_index() == 0:
            print(f"🔄 Cluster Restoration Active: Restoring state PyTree from step {latest_step}...")

        # 2. GENERATE THE TOPOLOGY-AGNOSTIC ABSTRACT REF
        # We extract an unallocated abstract template shape directly from our cached GraphDef blueprint
        def get_abstract_state_template():
            abstract_model = nnx.eval_shape(lambda: nnx.merge(self.graphdef, nnx.state(self.model)))
            _, abstract_state = nnx.split(abstract_model)
            return abstract_state

        abstract_template = get_abstract_state_template()

        # 3. RESTORE AND BIND TENSORS DIRECTLY BACK TO NNX WORKSPACE OBJECTS
        # StandardRestore now reshards and maps the stored array fragments onto any TPU layout seamlessly
        restored_state = self.mngr.restore(
            latest_step,
            args=ocp.args.StandardRestore(abstract_template)
        )

        # Merge the restored parameter values back into your live execution tensors in-place
        nnx.update((self.model, self.optimizer), restored_state)
        
        return int(latest_step)
