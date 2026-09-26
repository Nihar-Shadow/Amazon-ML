import logging
import sys
import time
import os
import psutil
from contextlib import contextmanager

def get_logger(name: str = "entity_resolution", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

def get_memory_usage_mb() -> float:
    """Returns current process RSS memory in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

@contextmanager
def log_step(step_name: str, logger: logging.Logger = None):
    """Context manager to log the duration and memory impact of an operation."""
    if logger is None:
        logger = get_logger()
        
    start_mem = get_memory_usage_mb()
    start_time = time.time()
    logger.info(f"START: {step_name} (Memory: {start_mem:.2f} MB)")
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        end_mem = get_memory_usage_mb()
        delta_mem = end_mem - start_mem
        logger.info(
            f"FINISHED: {step_name} in {elapsed:.2f}s | "
            f"End Mem: {end_mem:.2f} MB ({delta_mem:+.2f} MB)"
        )
