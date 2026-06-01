import os
import glob
import numpy as np
from roxxel import Roxxel

def clean_shards(base_name="test_sharded"):
    for file in glob.glob(f"{base_name}*"):
        os.remove(file)

def test_fused_sharded_mode():
    print("--- Testing Fused Sharded Block Compilation ---")
    base_name = "./test_sharded"
    clean_shards(base_name)

    # 1. Generate text sentences of variable lengths
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Generative AI is transforming sequence modeling.",
        "Xenron is a state space model with dynamic causal routing.",
        "Zero-RAM memory mapping is incredibly efficient."
    ]

    # 2. Write uniform 64-byte block records natively using Roxxel
    # Pattern will be dynamically resolved to a valid base file for writing!
    rox = Roxxel(filepath=f"{base_name}_*.rox")
    rox.write(sentences, block_size=64, max_shard_bytes=180, separator=b"\xff")

    # Verify shards were created on-disk
    created_shards = sorted(glob.glob(f"{base_name}_*.rox"))
    print(f"Created shards: {created_shards}")
    assert len(created_shards) > 1

    # 3. Read and stream using the unified stream API
    with Roxxel(filepath=f"{base_name}_*.rox") as dataset:
        print(f"Virtualized Block Dataset Length: {len(dataset)}")
        assert len(dataset[0]) == 64  # Every record is exactly 64 bytes!
        
        # Verify steps estimation helper
        assert dataset.estimate_steps(seq_len=32, batch_size=2) == 4
        
        # Test NumPy streaming of shape (batch_size=2, seq_len=32)
        stream = dataset.stream(seq_len=32, batch_size=2, seed=42)
        assert len(stream) == 4  # Expose exact step count!
        
        first_batch = next(stream)
        print(f"Batch Shape: {first_batch.shape}, dtype: {first_batch.dtype}")
        assert first_batch.shape == (2, 32)
        assert first_batch.dtype == np.int32

    # Clean up files
    clean_shards(base_name)
    print("Fused Sharded Block Compilation passed successfully!\n")

def test_int32_tokenized_dataset():
    print("--- Testing Tokenized int32 Dataset ---")
    base_name = "./test_tokenized"
    clean_shards(base_name)

    # 1. Generate fake token arrays (dtype=int32)
    # Each sequence has 16 tokens of 4 bytes each = 64 bytes
    fake_tokens = [
        np.arange(16, dtype=np.int32),
        np.arange(16, 32, dtype=np.int32),
        np.arange(32, 48, dtype=np.int32),
        np.arange(48, 64, dtype=np.int32),
    ]

    # 2. Write natively (dtype=int32 will be auto-detected!)
    rox = Roxxel(filepath=f"{base_name}_*.rox")
    rox.write(fake_tokens, block_size=64, max_shard_bytes=180, separator=None)

    # Verify shards were created
    created_shards = sorted(glob.glob(f"{base_name}_*.rox"))
    print(f"Created shards: {created_shards}")
    assert len(created_shards) > 1

    # 3. Read and stream natively
    with Roxxel(filepath=f"{base_name}_*.rox") as dataset:
        print(f"Virtualized Dataset Native Dtype: {dataset.dtype}")
        assert dataset.dtype == "int32"
        assert len(dataset) == 4
        
        # Verify steps estimation helper for multi-byte dtypes (2 * 8 * 4 = 64 bytes per batch)
        assert dataset.estimate_steps(seq_len=8, batch_size=2) == 4
        
        # Stream shape (batch_size=2, seq_len=8, dtype=int32)
        stream = dataset.stream(seq_len=8, batch_size=2, seed=42)
        assert len(stream) == 4  # Expose exact step count!
        
        first_batch = next(stream)
        print(f"Batch Shape: {first_batch.shape}, dtype: {first_batch.dtype}")
        assert first_batch.shape == (2, 8)
        assert first_batch.dtype == np.int32
        
        # Verify the actual token values are preserved and not corrupted!
        assert np.all(first_batch >= 0) and np.all(first_batch < 64)

    # Clean up files
    clean_shards(base_name)
    print("Tokenized int32 Dataset passed successfully!\n")

def test_logger():
    print("--- Testing Logger ---")
    from roxxel import Logger
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Test normal logging
        with Logger(log_dir=temp_dir, filename_prefix="test_log") as logger:
            logger.log_message("Hello from the test!")
        
        log_file = os.path.join(temp_dir, "test_log_system.log")
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            content = f.read()
            assert "Hello from the test!" in content
            
        # Test exception tracking and bubbling
        try:
            with Logger(log_dir=temp_dir, filename_prefix="test_crash") as logger:
                logger.log_message("About to crash...")
                raise ValueError("Oops, simulation crash!")
        except ValueError as e:
            assert str(e) == "Oops, simulation crash!"
            
        crash_log_file = os.path.join(temp_dir, "test_crash_system.log")
        assert os.path.exists(crash_log_file)
        with open(crash_log_file, "r") as f:
            content = f.read()
            assert "About to crash..." in content
            assert "CRITICAL: Uncaught exception occurred during execution!" in content
            assert "ValueError: Oops, simulation crash!" in content

    finally:
        shutil.rmtree(temp_dir)
    print("Logger tests passed successfully!\n")

if __name__ == "__main__":
    test_fused_sharded_mode()
    test_int32_tokenized_dataset()
    test_logger()
