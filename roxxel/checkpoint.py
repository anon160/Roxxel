import os
from flax import nnx
import orbax.checkpoint as ocp

class Checkpointer:
    """
    Asynchronous JAX/Flax NNX module and optimizer checkpointer.
    
    Uses Orbax Checkpoint Manager underneath to perform zero-overhead, multi-threaded
    state serialization on background threads, preventing disk writes from blocking 
    accelerator (GPU/TPU) training.
    
    Supports topology-agnostic PyTree reconstruction and automated best-loss tracking.
    """
    def __init__(self, checkpoint_path: str, model: nnx.Module, optimizer: nnx.Optimizer, max_to_keep: int = 3, timeout: int = 1000):
        """
        Args:
            checkpoint_path (str): The local or cloud storage path where checkpoints are written.
            model (flax.nnx.Module): The JAX/Flax NNX model state to serialize.
            optimizer (flax.nnx.Optimizer): The JAX/Flax NNX optimizer state containing optimizer parameters.
            max_to_keep (int, optional): The maximum number of recent checkpoints to retain. Defaults to 3.
            timeout (int, optional): The timeout in seconds for background asynchronous operations. Defaults to 1000.
        """
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.model = model
        self.optimizer = optimizer
        self.graphdef, _ = nnx.split((self.model, self.optimizer))
        
        async_opts = ocp.options.AsyncOptions(
            timeout_secs=timeout,                
            create_directories_asynchronously=True
        )

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep, 
            create=True,
            async_options=async_opts,
            best_fn=lambda metrics: metrics['loss'], 
            best_mode='min'                           
        )
        self.mngr = ocp.CheckpointManager(self.checkpoint_path, options=options)

    def save(self, step: int, metrics_dict: dict):
        """Extracts the global variable states and delegates storage and optimization metrics to Orbax.

        Args:
            step (int): The current training step number (used as the checkpoint subdirectory key).
            metrics_dict (dict): A dictionary containing metrics (like 'loss') to track best checkpoints.
        """
        _, state = nnx.split((self.model, self.optimizer))
        loss_val = float(metrics_dict.get("loss", 999.0))
        
        # Orbax handles background multi-threading, file validation, and rank-zero safely.
        self.mngr.save(
            int(step),
            args=ocp.args.StandardSave(state),
            metrics={'loss': loss_val}
        )

    def restore(self) -> int:
        """Restores parameters and optimizer tracking vectors natively without dictionary nesting.

        Returns:
            int: The step index of the restored checkpoint, or 0 if no checkpoint was found.
        """
        latest_step = self.mngr.latest_step()
        if latest_step is None:
            return 0

        def get_abstract_state_template():
            abstract_model = nnx.eval_shape(lambda: nnx.merge(self.graphdef, nnx.state(self.model)))
            _, abstract_state = nnx.split(abstract_model)
            return abstract_state

        abstract_template = get_abstract_state_template()

        restored_state = self.mngr.restore(
            latest_step,
            args=ocp.args.StandardRestore(abstract_template)
        )

        nnx.update((self.model, self.optimizer), restored_state)
        return int(latest_step)
