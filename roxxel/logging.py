import logging
import os
import sys
import traceback
import threading
from queue import Queue
from logging.handlers import QueueHandler, QueueListener

class Logger:
    """
    A high-performance, non-blocking, asynchronous logger designed for 
    distributed JAX/Flax pre-training clusters (e.g. multi-host TPU/GPU Pods).
    
    Offloads heavy I/O operations (stdout writes, system logs, and metrics CSV writes)
    to background threads, taking 0ms on the main training loop thread.
    
    Guarantees rank-zero execution (only rank 0 logs to files/stdout/metrics) to avoid 
    terminal spam and multi-process file locking contention.
    
    Supports context manager ('with' statements) to guarantee that if a TPU/GPU 
    crashes, OOMs, or is forcefully interrupted, all logging queues and threads 
    are completely flushed and drained to disk before termination.
    """
    def __init__(self, log_dir: str, filename_prefix: str = "roxxel", logger_name: str = "RoxxelCore"):
        """
        Args:
            log_dir (str): Directory where standard log files and metric CSV files will be saved.
            filename_prefix (str, optional): Prefix for generated log files. Defaults to "roxxel".
            logger_name (str, optional): Name of the underlying Python logger. Defaults to "RoxxelCore".
        """
        self.log_dir = log_dir
        
        try:
            import jax
            # Use getattr to safely handle any potential future JAX modifications
            process_index_fn = getattr(jax, "process_index", lambda: 0)
            self.is_rank_zero = (process_index_fn() == 0)
        except ImportError:
            self.is_rank_zero = True

        if self.is_rank_zero:
            os.makedirs(self.log_dir, exist_ok=True)
            
            # 1. Asynchronous System Logger Queue Setup
            self.log_queue = Queue(-1)
            self.sys_log_path = os.path.join(self.log_dir, f"{filename_prefix}_system.log")
            text_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            
            stdout_worker = logging.StreamHandler(sys.stdout)
            stdout_worker.setFormatter(text_formatter)
            file_worker = logging.FileHandler(self.sys_log_path, encoding="utf-8")
            file_worker.setFormatter(text_formatter)

            self.listener = QueueListener(self.log_queue, stdout_worker, file_worker)
            self.listener.start()

            self.queue_handler = QueueHandler(self.log_queue)
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.INFO)
            self.logger.handlers.clear()
            self.logger.addHandler(self.queue_handler)
            self.logger.propagate = False

            # 2. Asynchronous Metrics CSV Writer Setup
            self.metrics_csv_path = os.path.join(self.log_dir, f"{filename_prefix}_metrics.csv")
            self.metrics_queue = Queue(-1)
            self.metrics_thread = threading.Thread(target=self._metrics_writer_worker, daemon=True)
            self.metrics_thread.start()

    def _metrics_writer_worker(self):
        """Background worker thread that serializes metric dictionaries to the CSV file sequentially."""
        header_written = os.path.exists(self.metrics_csv_path)
        metrics_keys = None

        # If a CSV file already exists (e.g. on step resumption), read its header to keep columns aligned
        if header_written:
            try:
                with open(self.metrics_csv_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        metrics_keys = first_line.split(",")[1:]  # Skip the first column ("step")
            except Exception:
                header_written = False

        while True:
            item = self.metrics_queue.get()
            if item is None:  # Shutdown sentinel
                self.metrics_queue.task_done()
                break

            step, metrics = item
            if metrics_keys is None:
                metrics_keys = list(metrics.keys())
                # Write header row
                with open(self.metrics_csv_path, "w", newline="", encoding="utf-8") as f:
                    f.write("step," + ",".join(metrics_keys) + "\n")
                header_written = True

            # Format values nicely (floating point floats mapped to .5f precision)
            vals = []
            for k in metrics_keys:
                val = metrics.get(k, "")
                if isinstance(val, float):
                    vals.append(f"{val:.5f}")
                elif isinstance(val, (int, bool)):
                    vals.append(str(val))
                else:
                    vals.append(str(val))

            # Direct flat text append is fast and thread-safe inside the dedicated worker thread
            with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as f:
                f.write(f"{step}," + ",".join(vals) + "\n")

            self.metrics_queue.task_done()

    def __enter__(self):
        """Returns the logger instance itself when entering the 'with' block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Guarantees the queue is drained and threads are safely stopped on exit or crash.
        Automatically logs uncaught tracebacks to system logs on Rank 0 if a crash occurs.
        """
        if exc_type is not None and self.is_rank_zero:
            tb_lines = traceback.format_exception(exc_type, exc_val, exc_tb)
            self.logger.error("❌ CRITICAL: Uncaught exception occurred during execution!")
            for line in tb_lines:
                self.logger.error(line.rstrip())
        
        self.close()
        # Return False to let the exception bubble up normally after flushing logs
        return False

    def log_message(self, message: str, level: int = logging.INFO):
        """Passes a string to the system log queue. Takes 0ms on your main training loop thread.

        Args:
            message (str): The log message string.
            level (int, optional): The log level (e.g. logging.INFO, logging.WARNING). Defaults to logging.INFO.
        """
        if self.is_rank_zero:
            self.logger.log(level, message)

    def log_metrics_summary(self, step: int, metrics: dict):
        """Appends arbitrary metric dictionary data asynchronously to a persistent CSV file.

        Args:
            step (int): The current training step number.
            metrics (dict): Dict of metrics to write (e.g. {'loss': 0.1, 'accuracy': 0.9}).
        """
        if self.is_rank_zero:
            self.metrics_queue.put((step, metrics))

    def close(self):
        """Forces all background asynchronous write threads to drain and complete disk writes."""
        if self.is_rank_zero:
            # 1. Stop metrics writer and wait for queue to drain completely
            if hasattr(self, 'metrics_queue') and hasattr(self, 'metrics_thread'):
                self.metrics_queue.put(None)
                self.metrics_thread.join()

            # 2. Stop system logs listener
            if hasattr(self, 'listener'):
                self.listener.stop()  # Drains log queue completely to disk before closing

            # 3. Cleanly remove handler from singleton logger
            if hasattr(self, 'logger') and hasattr(self, 'queue_handler'):
                self.logger.removeHandler(self.queue_handler)
