import os
import sys
import traceback

class Logger:
    """
    A lightweight, asynchronous-friendly Rank-0 Logger for Roxxel.
    
    Provides interactive terminal progress bars via `tqdm` and optional 
    cloud-based run tracking via `wandb` (Weights & Biases). Coordinates 
    stdout printing exclusively on Rank 0 to prevent terminal clutter in 
    distributed JAX/Flax training runs.
    """
    def __init__(self, log_dir: str = None, project: str = None, name: str = None, config: dict = None):
        """
        Args:
            log_dir (str, optional): Root directory to save logs (no-op here, kept for backward compatibility).
            project (str, optional): Weights & Biases project name. If provided, initializes wandb.
            name (str, optional): Display name for the Weights & Biases run.
            config (dict, optional): Hyperparameter dictionary to save to the wandb run configuration.
        """
        import jax
        self.is_rank_zero = (jax.process_index() == 0)
        self.project = project
        self.pbar = None
        self._log_dir = log_dir

        if self.is_rank_zero:
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            if project:
                import wandb
                wandb.init(project=project, name=name, config=config)

    def init_pbar(self, total_steps: int, initial_step: int = 0):
        """
        Lazily initializes the tqdm progress bar on Rank 0 once the 
        total optimization horizon is determined.
        """
        if self.is_rank_zero and self.pbar is None:
            from tqdm.auto import tqdm
            self.pbar = tqdm(total=total_steps, initial=initial_step, desc="Training")

    def update_pbar(self, step: int):
        """Updates the progress bar step count on Rank 0."""
        if self.is_rank_zero and self.pbar is not None:
            self.pbar.update(step - self.pbar.n)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_rank_zero:
            if exc_type is not None:
                # Intercept crash tracebacks to log them cleanly to stderr
                import traceback
                print("\n❌ CRITICAL: Uncaught exception occurred during execution!", file=sys.stderr)
                traceback.print_exception(exc_type, exc_val, exc_tb, file=sys.stderr)
                
                if self.project:
                    import wandb
                    wandb.finish(exit_code=1)
            else:
                self.close()
        return False

    def log_message(self, message: str, level: int = None):
        """Prints a message to stdout on Rank 0, pushing it safely above the active progress bar."""
        if self.is_rank_zero:
            if self.pbar is not None:
                from tqdm.auto import tqdm
                tqdm.write(message)
            else:
                print(message)

    def log_metrics_summary(self, step: int, metrics: dict):
        """Updates the progress bar postfix metrics and pushes summaries asynchronously to WandB."""
        if self.is_rank_zero:
            if self.pbar is not None:
                self.pbar.update(step - self.pbar.n)
                # Format floating points to avoid float representation clutter in terminal
                formatted_metrics = {
                    k: f"{v:.4f}" if isinstance(v, float) else str(v)
                    for k, v in metrics.items()
                }
                self.pbar.set_postfix(**formatted_metrics)
            if self.project:
                import wandb
                wandb.log(metrics, step=step)

    def close(self):
        """Cleans up and finalizes progress bars and wandb runs."""
        if self.is_rank_zero:
            if self.pbar is not None:
                self.pbar.close()
                self.pbar = None
            if self.project:
                import wandb
                wandb.finish()
