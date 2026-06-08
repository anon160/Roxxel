import os
import jax
import jax.numpy as jnp
from flax import nnx
from roxxel.core import Roxxel, RoxxelStream

class Phase:
    """
    Represents a single phase within a training curriculum schedule.

    Attributes:
        steps (int): The target number of training steps for this phase.
        batch_size (int): The batch size to yield during this phase.
        seq_len (int): The sequence length of each sample in the batch.
        weights (dict, optional): Dataset blending weights mapping dataset keys to float ratios.
            Must match the keys provided in Curriculum.mix_streamers (along with 'self').
    """
    def __init__(self, steps: int, batch_size: int, seq_len: int, weights: dict = None):
        self.steps = steps
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.weights = weights

class Curriculum:
    """
    Manages the multi-phase training curriculum and dataset blending streams.

    Attributes:
        primary_streamer (Roxxel): The primary dataset streamer instance.
        phases (list of Phase): The curriculum timeline of training phases.
        mix_streamers (dict of str: Roxxel, optional): Secondary datasets to mix.
    """
    def __init__(self, primary_streamer: Roxxel, phases: list[Phase], mix_streamers: dict[str, Roxxel] = None):
        self.primary_streamer = primary_streamer
        self.phases = phases
        self.mix_streamers = mix_streamers

class ModelState(nnx.Module):
    """
    Unified JAX/Flax NNX state module containing the model, optimizer,
    and step counter. Created internally by the Trainer.
    """
    def __init__(self, model: nnx.Module, optimizer: nnx.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.step = nnx.Variable(jnp.array(0, dtype=jnp.int32))

class Trainer:
    """
    Curriculum-aware pre-training orchestrator designed for JAX/Flax NNX.
    
    Accepts the Curriculum schedule (which wraps the Roxxel dataset streamers) and
    manages the pre-training loop execution, boundary transitions, hot-swapping,
    asynchronous logging, evaluations, and Orbax checkpointing.
    """
    def __init__(
        self,
        model,
        optimizer,
        curriculum: Curriculum,
        loss_fn,
        save_path=None,
        checkpointer=None,
        logger=None,
        eval_fn=None,
        eval_every: int = 500,
        checkpoint_every: int = 100,
        log_every: int = 100,
        seed: int = 42,
        mesh=None,
        data_sharding=None,
        max_to_keep: int = 3,
        timeout: int = 1000,
    ):
        """
        Args:
            model (flax.nnx.Module): The JAX training state / model instance. If a pre-constructed state
                object containing `model` and `optimizer` attributes is passed, the trainer
                automatically detects it for backward compatibility.
            optimizer (flax.nnx.Optimizer): The Optax optimizer/Flax NNX optimizer instance. Can be None if a
                pre-constructed state is passed as the first argument.
            curriculum (Curriculum): The curriculum schedule object.
            loss_fn (callable): The loss function: loss_fn(model, batch) -> scalar or tuple (loss, aux).
            save_path (str, optional): The root directory where checkpoints and logs are saved.
                If provided, `checkpointer` defaults to `save_path/checkpoints` and `logger`
                defaults to `save_path`.
            checkpointer (Checkpointer, str, optional): Asynchronous Checkpointer instance
                or directory path to automatically initialize it.
            logger (Logger, str, optional): Asynchronous Logger instance or directory
                path to automatically initialize it.
            eval_fn (callable, optional): Callback for periodic evaluations: eval_fn(state) -> str/None.
            eval_every (int, optional): Run evaluations every N steps. Defaults to 500.
            checkpoint_every (int, optional): Save checkpoint every N steps. Defaults to 100.
            log_every (int, optional): Log training metrics every N steps. Defaults to 100.
            seed (int, optional): Base random seed for stream replication. Defaults to 42.
            mesh (jax.sharding.Mesh, optional): JAX hardware mesh sharding specification.
            data_sharding (jax.sharding.NamedSharding, optional): JAX named sharding specification.
            max_to_keep (int, optional): Max checkpoints to keep when initializing checkpointer path. Defaults to 3.
            timeout (int, optional): Timeout for async operations when initializing checkpointer path. Defaults to 1000.
        """
        # Check if the first parameter is actually a state object
        if hasattr(model, "model") and hasattr(model, "optimizer"):
            self.state = model
            self.model = model.model
            self.optimizer = optimizer if optimizer is not None else model.optimizer
        else:
            self.model = model
            self.optimizer = optimizer
            self.state = ModelState(model, optimizer)

        self.curriculum = curriculum
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.eval_every = eval_every
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        self.seed = seed
        self.mesh = mesh
        self.data_sharding = data_sharding

        # Merge checkpointer and logger if save_path is provided
        if save_path is not None:
            if checkpointer is None:
                checkpointer = os.path.join(save_path, "checkpoints")
            if logger is None:
                logger = save_path

        # Handle Checkpointer initialization
        self._own_checkpointer = False
        if isinstance(checkpointer, str):
            from roxxel.checkpoint import Checkpointer
            self.checkpointer = Checkpointer(
                checkpoint_path=checkpointer,
                model=self.model,
                optimizer=self.optimizer,
                max_to_keep=max_to_keep,
                timeout=timeout
            )
            self._own_checkpointer = True
        else:
            self.checkpointer = checkpointer

        # Handle Logger initialization
        self._own_logger = False
        if isinstance(logger, str):
            from roxxel.logging import Logger
            self.logger = Logger(log_dir=logger)
            self._own_logger = True
        else:
            self.logger = logger

        # Build and JIT compile the training step internally
        @nnx.jit
        def train_step(state, batch):
            def loss_wrapper(model):
                out = self.loss_fn(model, batch)
                # Ensure only the scalar loss is returned for gradients
                if isinstance(out, (tuple, list)):
                    return out[0]
                elif isinstance(out, dict):
                    if "loss" in out:
                        return out["loss"]
                    return next(iter(out.values()))
                return out
                
            loss, grads = nnx.value_and_grad(loss_wrapper)(state.model)
            try:
                state.optimizer.update(state.model, grads)
            except TypeError:
                state.optimizer.update(grads)
            try:
                state.step[...] += 1
            except (TypeError, ValueError, AttributeError):
                state.step.value += 1
            return {"loss": loss, "ppl": jnp.exp(loss)}
            
        self.train_step_fn = train_step

    def run(self):
        """
        Executes the curriculum training loop, automatically handling skips, resumptions,
        blending weights, and dynamic shape transitions at phase boundaries.
        """
        if self._own_logger and self.logger:
            with self.logger:
                self._run()
        else:
            self._run()

    def _run(self):
        # 1. Restore checkpoints if available
        start_step = 0
        if self.checkpointer:
            start_step = self.checkpointer.restore()
            
        # Update JAX-state step counter
        if hasattr(self.state, "step"):
            try:
                self.state.step[...] = jnp.array(start_step, dtype=jnp.int32)
            except (TypeError, ValueError, AttributeError):
                if hasattr(self.state.step, "value"):
                    self.state.step.value = jnp.array(start_step, dtype=jnp.int32)
            
        # 2. Determine initial curriculum configuration window
        accumulated_steps = 0
        completed_phases_ledger = []
        
        current_seq_len = None
        current_batch_size = None
        current_phase_total_steps = None
        current_weights = None
        
        for idx, phase in enumerate(self.curriculum.phases):
            p_steps = phase.steps
            p_batch = phase.batch_size
            p_seq = phase.seq_len
            p_weights = phase.weights
            
            if start_step >= accumulated_steps + p_steps:
                # Add fully completed phases to the historical ledger
                completed_phases_ledger.append((p_steps, p_batch, p_seq))
                accumulated_steps += p_steps
            else:
                # Active phase configuration branch located
                current_batch_size = p_batch
                current_seq_len = p_seq
                current_phase_total_steps = p_steps
                current_weights = p_weights
                break
                
        # Calculate remaining target steps for the active streaming window session
        steps_already_done_in_current_phase = start_step - accumulated_steps
        remaining_steps_for_session = current_phase_total_steps - steps_already_done_in_current_phase
        
        total_train_steps = sum(p.steps for p in self.curriculum.phases)
        
        if self.logger:
            self.logger.log_message(f"🎯 Total Optimization Horizon: {total_train_steps} global steps.")
            self.logger.log_message(f"♻️ Resuming active phase layout: [SEQ: {current_seq_len} | BATCH: {current_batch_size}]")
            self.logger.log_message(f"📊 Remaining steps for this configuration window: {remaining_steps_for_session}")

        dataset = self.curriculum.primary_streamer
        if not dataset._is_open:
            dataset.open()
            
        try:
            # Helper function to construct sharded dataset streams
            def make_stream(seq_len, batch_size, step, ledger, steps_limit, weights):
                return dataset.stream(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    seed=self.seed,
                    start_step=step,
                    completed_phases=ledger,
                    total_steps=steps_limit,
                    mesh=self.mesh,
                    data_sharding=self.data_sharding,
                    mix_datasets=self.curriculum.mix_streamers,
                    weights=weights
                )
                
            loader_stream = make_stream(
                current_seq_len,
                current_batch_size,
                start_step,
                completed_phases_ledger,
                remaining_steps_for_session,
                current_weights
            )
            
            curr_step = start_step
            while curr_step < total_train_steps:
                for batch in loader_stream:
                    metrics = self.train_step_fn(self.state, batch)
                    
                    if hasattr(self.state, "step"):
                        try:
                            curr_step = int(self.state.step.value)
                        except (TypeError, ValueError, AttributeError):
                            try:
                                curr_step = int(self.state.step)
                            except (TypeError, ValueError, AttributeError):
                                curr_step += 1
                    else:
                        curr_step += 1
                        
                    # 1. Asynchronous system logging
                    if curr_step % self.log_every == 0 and self.logger:
                        loss_val = float(metrics["loss"])
                        ppl = float(metrics["ppl"])
                        self.logger.log_message(f"S{curr_step} | Loss: {loss_val:.4f} | PPL: {ppl:.2f}")
                        self.logger.log_metrics_summary(step=curr_step, metrics={"loss": loss_val, "perplexity": ppl})
                        
                    # 2. Asynchronous checkpointing
                    if curr_step % self.checkpoint_every == 0 and self.checkpointer:
                        self.checkpointer.save(curr_step, metrics_dict={"loss": float(metrics["loss"])})
                        
                    # 3. Model sampling/evaluation
                    if self.eval_fn and curr_step % self.eval_every == 0:
                        if self.logger:
                            self.logger.log_message(f"🧪 Running Evaluation Check at Step {curr_step}...")
                        story = self.eval_fn(self.state)
                        if self.logger and story:
                            self.logger.log_message(f"EVALUATION OUTPUT:\n{story}\n")
                            
                    # 4. Extensible phase transition swap
                    phase_boundary_accumulator = 0
                    for phase_idx, phase in enumerate(self.curriculum.phases[:-1]):
                        phase_boundary_accumulator += phase.steps
                        
                        if curr_step == phase_boundary_accumulator:
                            next_phase = self.curriculum.phases[phase_idx + 1]
                            next_steps = next_phase.steps
                            next_batch = next_phase.batch_size
                            next_seq = next_phase.seq_len
                            next_weights = next_phase.weights
                            
                            if self.logger:
                                self.logger.log_message(f"🎯 Step {curr_step} hit! Swapping dynamically to Phase {phase_idx + 2} [SEQ: {next_seq} | BATCH: {next_batch}]...")
                                
                            # Expand historical ledger
                            completed_phases_ledger = [
                                (p.steps, p.batch_size, p.seq_len)
                                for p in self.curriculum.phases[:phase_idx + 1]
                            ]
                            
                            # Re-instantiate JAX stream with updated shape configurations
                            loader_stream = make_stream(
                                next_seq,
                                next_batch,
                                curr_step,
                                completed_phases_ledger,
                                next_steps,
                                next_weights
                            )
                            break
                            
                    if curr_step >= total_train_steps:
                        if self.logger:
                            self.logger.log_message(f"🏁 Curriculum complete: {curr_step}/{total_train_steps} steps finished successfully.")
                        break
        finally:
            dataset.close()
            if self._own_checkpointer and self.checkpointer:
                # Wait for any pending async checkpoint saves to finish
                if hasattr(self.checkpointer, "mngr") and hasattr(self.checkpointer.mngr, "wait_until_finished"):
                    try:
                        self.checkpointer.mngr.wait_until_finished()
                    except Exception:
                        pass
                if hasattr(self.checkpointer, "mngr") and hasattr(self.checkpointer.mngr, "close"):
                    try:
                        self.checkpointer.mngr.close()
                    except Exception:
                        pass
            if self.logger:
                self.logger.log_message("✅ Global Multi-Phase Execution Complete. Roxxel Instance Closed Safely.")
