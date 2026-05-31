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
        
        # Test NumPy streaming of shape (batch_size=2, seq_len=32)
        stream = dataset.stream(seq_len=32, batch_size=2, seed=42)
        
        first_batch = next(stream)
        print(f"Batch Shape: {first_batch.shape}, dtype: {first_batch.dtype}")
        assert first_batch.shape == (2, 32)
        assert first_batch.dtype == np.int32

    # Clean up files
    clean_shards(base_name)
    print("Fused Sharded Block Compilation passed successfully!\n")

if __name__ == "__main__":
    test_fused_sharded_mode()
