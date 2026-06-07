# Logging

Roxxel's `Logger` implements a high-performance, queue-based logging architecture. To prevent system I/O (like writes to stdout, files, and CSVs) from slowing down high-throughput TPU/GPU training loops, all log writing is offloaded to background worker threads.

---

## Why the Context Manager is Critical

Because logging happens asynchronously on a separate background thread, standard print statements or un-managed logs are highly vulnerable. If your TPU/GPU throws an Out of Memory (OOM) error or JAX crashes:
1. The main thread terminates instantly.
2. The logging queue gets cut off.
3. The most critical debug messages/tracebacks at the end of the run are lost.

Using the **`with`** context manager solves this completely:

```python
import time
from roxxel import Logger

# Initialize the async logger context
with Logger(log_dir="./run_directory") as logger:
    logger.log_message("Initializing deep pre-training cluster...")
    
    # Under the hood, any exceptions raised here are caught by the context manager.
    # It logs the traceback, drains/flushes the async queue to disk, and then propagates the error.
    time.sleep(1)
    raise RuntimeError("TPU Device Out of Memory!")

# The background thread is safely joined and shut down here.
```

### Automatic Crash Traceback Capture
When a crash occurs inside the `Logger` context:
1. The traceback is immediately intercepted.
2. It writes the traceback cleanly to both stdout and `{log_dir}/{prefix}_system.log`.
3. It forces the queue to block and drain entirely, guaranteeing that **every single log line is written to disk before the program terminates**.

---

## Multi-Host TPU Rank-Zero Filter

When scaling JAX code across TPU Pods or multi-node GPU clusters, standard print statements are executed by every worker node simultaneously, resulting in corrupted, duplicate log files.

Roxxel's `Logger` detects JAX rank automatically:
* **Only Rank 0** writes messages to stdout, log files, or CSVs.
* **Other ranks (1..N)** execute logging statements as safe `noop` operations, preventing file conflicts and terminal pollution.

---

## Asynchronous Metrics CSV Logging

You can record training metrics (like loss, learning rate, and perplexity) directly to a CSV file without blocking JAX JIT execution:

```python
from roxxel import Logger

with Logger(log_dir="./logs", filename_prefix="run_alpha") as logger:
    for step in range(100):
        # Your JAX training loop here
        loss = 2.5 - (step * 0.01)
        lr = 3e-4
        
        # This pushes metrics to a background queue instantly (0ms overhead)
        logger.log_metrics_summary(
            step=step,
            metrics={"loss": loss, "lr": lr}
        )
```
This produces `logs/run_alpha_metrics.csv` automatically with properly aligned column headers on step resumption.

---

## API Reference

::: roxxel.logging.Logger
    options:
      show_root_heading: true
