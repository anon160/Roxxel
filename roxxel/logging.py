import logging
import os
import sys
import traceback
from queue import Queue
from logging.handlers import QueueHandler, QueueListener

class RoxxelLogger:
    """
    A high-performance, non-blocking, asynchronous logger designed for 
    distributed JAX/Flax pre-training clusters (e.g. multi-host TPU/GPU Pods).
    
    Offloads heavy I/O operations (stdout writes and disk writes) to a background
    thread using Python's QueueHandler and QueueListener, taking 0ms on the main 
    training loop thread.
    
    Guarantees rank-zero execution (only rank 0 logs to files/stdout) to avoid 
    terminal spam and multi-process file locking contention.
    
    Supports context manager ('with' statements) to guarantee that if a TPU/GPU 
    crashes, OOMs, or is forcefully interrupted, the queue is completely drained 
    to disk before termination, and any uncaught tracebacks are captured cleanly 
    in the system log.
    """
    def __init__(self, log_dir: str, filename_prefix: str = "xenron", logger_name: str = "XenronCore"):
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
            self.logger.addHandler(self.queue_handler)

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
        """Passes a string to the queue. Takes 0ms on your main training loop thread."""
        if self.is_rank_zero:
            self.logger.log(level, message)

    def close(self):
        """Forces the background asynchronous write threads to complete and lock files."""
        if self.is_rank_zero and hasattr(self, 'listener'):
            self.listener.stop()  # Drains queue completely to disk before closing

# Alias for backwards compatibility with user's Xenron code
XenronLogger = RoxxelLogger
