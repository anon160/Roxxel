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
        # Test normal logging and metrics
        with Logger(log_dir=temp_dir, filename_prefix="test_log") as logger:
            logger.log_message("Hello from the test!")
            logger.log_metrics_summary(step=10, metrics={"loss": 1.23456, "perplexity": 3.456})
        
        log_file = os.path.join(temp_dir, "test_log_system.log")
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            content = f.read()
            assert "Hello from the test!" in content

        csv_file = os.path.join(temp_dir, "test_log_metrics.csv")
        assert os.path.exists(csv_file)
        with open(csv_file, "r") as f:
            lines = f.readlines()
            assert lines[0] == "step,loss,perplexity\n"
            assert lines[1] == "10,1.23456,3.45600\n"
            
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

def test_incomplete_shard_recovery():
    print("--- Testing Incomplete Shard Recovery (Errno 22 Fix) ---")
    base_name = "./test_incomplete"
    clean_shards(base_name)
    
    # 1. Create a corrupted/empty shard file of 5 bytes (size < 24)
    corrupted_shard = f"{base_name}_0000.rox"
    with open(corrupted_shard, "wb") as f:
        f.write(b"hello")
        
    # 2. Try writing to this sharded path. Roxxel should successfully reuse/overwrite the file 
    # without raising OSError [Errno 22] Invalid argument.
    # Instantiate with the base path, not the wildcard!
    rox = Roxxel(filepath=f"{base_name}.rox")
    rox.write(["Test sentence to overwrite the corrupt file."], block_size=64, max_shard_bytes=100, separator=None)
    
    # 3. Read it back using a wildcard pattern and assert it is valid
    with Roxxel(filepath=f"{base_name}_*.rox") as dataset:
        assert len(dataset) == 1
        assert len(dataset[0]) == 64
        
    clean_shards(base_name)
    print("Incomplete Shard Recovery test passed successfully!\n")

def test_multiphase_curriculum_stream():
    print("--- Testing Multiphase Curriculum Stream Resumption ---")
    base_name = "./test_curriculum"
    clean_shards(base_name)

    # 1. Write a sequence of int32 tokens: 16 blocks of 64 tokens = 1024 tokens.
    fake_tokens = [
        np.arange(i * 64, (i + 1) * 64, dtype=np.int32)
        for i in range(16)
    ]
    
    rox = Roxxel(filepath=f"{base_name}_*.rox")
    # Write to a single or sharded file
    rox.write(fake_tokens, block_size=256, max_shard_bytes=10000, separator=None)

    with Roxxel(filepath=f"{base_name}_*.rox") as dataset:
        # Load all tokens sequentially in shuffled block order to compare
        total_blocks = len(dataset)
        global_indices = np.arange(total_blocks)
        rng = np.random.default_rng(42)
        rng.shuffle(global_indices)
        
        flat_shuffled_bytes = np.concatenate([dataset[int(idx)] for idx in global_indices])
        flat_shuffled = np.frombuffer(flat_shuffled_bytes.tobytes(), dtype=np.dtype(dataset.dtype))
        
        # Define the curriculum:
        # Phase 0: 5 steps, batch_size=4, seq_len=8 (total 160 tokens)
        # Phase 1: 3 steps, batch_size=2, seq_len=16 (total 96 tokens)
        # Phase 2: 2 steps, batch_size=8, seq_len=4 (total 64 tokens)
        
        completed_phases = [
            (5, 4, 8),    # Phase 0: 5 steps, batch_size 4, seq_len 8
            (3, 2, 16),   # Phase 1: 3 steps, batch_size 2, seq_len 16
        ]
        
        # Test Case A: Streaming Phase 0 (start_step = 0, no completed phases)
        stream_p0 = dataset.stream(seq_len=8, batch_size=4, seed=42, start_step=0, total_steps=5)
        assert len(stream_p0) == 5
        batches_p0 = list(stream_p0)
        assert len(batches_p0) == 5
        assert batches_p0[0].shape == (4, 8)
        # Verify the tokens match flat_shuffled
        p0_reconstructed = np.concatenate([b.flatten() for b in batches_p0])
        assert np.array_equal(p0_reconstructed, flat_shuffled[0:160])
        
        # Test Case B: Resuming Phase 1 at step 5 (completed Phase 0)
        stream_p1 = dataset.stream(
            seq_len=16, batch_size=2, seed=42, start_step=5,
            completed_phases=[(5, 4, 8)], total_steps=3
        )
        assert len(stream_p1) == 3
        batches_p1 = list(stream_p1)
        assert len(batches_p1) == 3
        assert batches_p1[0].shape == (2, 16)
        p1_reconstructed = np.concatenate([b.flatten() for b in batches_p1])
        assert np.array_equal(p1_reconstructed, flat_shuffled[160:256])
        
        # Test Case C: Resuming Phase 2 at step 8 (completed Phase 0 and Phase 1)
        stream_p2 = dataset.stream(
            seq_len=4, batch_size=8, seed=42, start_step=8,
            completed_phases=completed_phases, total_steps=2
        )
        assert len(stream_p2) == 2
        batches_p2 = list(stream_p2)
        assert len(batches_p2) == 2
        assert batches_p2[0].shape == (8, 4)
        p2_reconstructed = np.concatenate([b.flatten() for b in batches_p2])
        assert np.array_equal(p2_reconstructed, flat_shuffled[256:320])

        # Test Case D: Resuming Phase 1 in the middle (e.g. start_step = 6, completed_phases=completed_phases, but we want 2 steps remaining)
        stream_p1_mid = dataset.stream(
            seq_len=16, batch_size=2, seed=42, start_step=6,
            completed_phases=completed_phases, total_steps=2
        )
        assert len(stream_p1_mid) == 2
        batches_p1_mid = list(stream_p1_mid)
        assert len(batches_p1_mid) == 2
        p1_mid_reconstructed = np.concatenate([b.flatten() for b in batches_p1_mid])
        assert np.array_equal(p1_mid_reconstructed, flat_shuffled[192:256])

    clean_shards(base_name)
    print("Multiphase Curriculum Stream Resumption passed successfully!\n")

def test_mixer_multiphase_stream():
    print("--- Testing Unified Roxxel.stream Mixing with Multiphase Resumption ---")
    import jax

    base_name1 = "./test_mixer_ds1"
    base_name2 = "./test_mixer_ds2"
    clean_shards(base_name1)
    clean_shards(base_name2)

    # 1. Create two distinct datasets
    tokens_ds1 = [np.arange(i * 64, (i + 1) * 64, dtype=np.int32) for i in range(8)]
    tokens_ds2 = [np.arange(1000 + i * 64, 1000 + (i + 1) * 64, dtype=np.int32) for i in range(8)]

    rox1 = Roxxel(filepath=f"{base_name1}_*.rox")
    rox1.write(tokens_ds1, block_size=256, max_shard_bytes=10000, separator=None)
    
    rox2 = Roxxel(filepath=f"{base_name2}_*.rox")
    rox2.write(tokens_ds2, block_size=256, max_shard_bytes=10000, separator=None)

    # Load datasets
    with Roxxel(filepath=f"{base_name1}_*.rox") as ds1, Roxxel(filepath=f"{base_name2}_*.rox") as ds2:
        # Define weights and datasets mapping
        mix_datasets = {"ds2": ds2}
        weights = {"self": 0.3, "ds2": 0.7}

        # Draw global choice simulation to verify manually
        seed = 42
        probs = [0.3, 0.7]
        rng = np.random.default_rng(seed)
        
        # Max steps upper bound:
        # ds1 (self) has 8 * 64 = 512 tokens. At batch=4, seq=8 (32 tokens/step), individual max is 16 steps.
        # ds2 has 512 tokens. Individual max is 16 steps.
        # Total global steps bound: 16 + 16 = 32 steps.
        # Plus any completed phase steps.
        
        # Let's stream from scratch: Phase 0 (5 steps, batch=4, seq=8)
        stream_p0 = ds1.stream(
            seq_len=8, batch_size=4, seed=seed, start_step=0, total_steps=5,
            mix_datasets=mix_datasets, weights=weights
        )
        assert len(stream_p0) == 5
        batches_p0 = list(stream_p0)
        assert len(batches_p0) == 5
        assert isinstance(batches_p0[0], jax.Array)
        assert batches_p0[0].shape == (4, 8)

        # Convert back to numpy to inspect values
        np_batches_p0 = [np.array(b) for b in batches_p0]
        
        # Resimulate choices for Phase 0
        all_choices = rng.choice(2, size=32 + 5, p=probs)  # 32 is max_possible_global_steps
        p0_choices = all_choices[0:5]
        
        # Verify choices
        # Let's get independent local streams using seed+i+1
        # seed for ds1 (self): seed + 1 = 43
        # seed for ds2: seed + 2 = 44
        counts_p0 = np.bincount(p0_choices, minlength=2)
        local_stream_ds1 = ds1.stream(seq_len=8, batch_size=4, seed=43, total_steps=counts_p0[0])
        local_stream_ds2 = ds2.stream(seq_len=8, batch_size=4, seed=44, total_steps=counts_p0[1])
        
        expected_batches = []
        for choice in p0_choices:
            if choice == 0:
                expected_batches.append(np.array(next(local_stream_ds1)))
            else:
                expected_batches.append(np.array(next(local_stream_ds2)))

        for b_act, b_exp in zip(np_batches_p0, expected_batches):
            assert np.array_equal(b_act, b_exp)

        # Test Case: Resume Phase 1 at step 5
        # Completed phases: [(5, 4, 8)]
        # Phase 1: 3 steps, batch_size=2, seq_len=16
        stream_p1 = ds1.stream(
            seq_len=16, batch_size=2, seed=seed, start_step=5,
            completed_phases=[(5, 4, 8)], total_steps=3,
            mix_datasets=mix_datasets, weights=weights
        )
        assert len(stream_p1) == 3
        batches_p1 = list(stream_p1)
        assert len(batches_p1) == 3
        
        # Resimulate local completed phase and start step allocation
        # Phase 0: 5 steps
        counts_p0 = np.bincount(all_choices[0:5], minlength=2)
        # Current step starts at 5, so no steps in current phase are simulated before start_step
        # Local start steps = 0 for the current phase (since start_step == global_accumulator)
        # So we expect:
        # local_completed_phases: self: [(counts_p0[0], 4, 8)], ds2: [(counts_p0[1], 4, 8)]
        # local_start_steps: self: 0, ds2: 0
        # Stream Phase 1: choices are all_choices[5:8]
        p1_choices = all_choices[5:8]
        counts_p1 = np.bincount(p1_choices, minlength=2)
        
        local_stream_p1_ds1 = ds1.stream(
            seq_len=16, batch_size=2, seed=43, start_step=0,
            completed_phases=[(counts_p0[0], 4, 8)], total_steps=counts_p1[0]
        )
        local_stream_p1_ds2 = ds2.stream(
            seq_len=16, batch_size=2, seed=44, start_step=0,
            completed_phases=[(counts_p0[1], 4, 8)], total_steps=counts_p1[1]
        )
        
        expected_batches_p1 = []
        for choice in p1_choices:
            if choice == 0:
                expected_batches_p1.append(np.array(next(local_stream_p1_ds1)))
            else:
                expected_batches_p1.append(np.array(next(local_stream_p1_ds2)))

        for b_act, b_exp in zip(batches_p1, expected_batches_p1):
            assert np.array_equal(np.array(b_act), b_exp)

    clean_shards(base_name1)
    clean_shards(base_name2)
    print("Unified Roxxel.stream mixing with Multiphase Resumption passed successfully!\n")

def test_filepath_auto_resolution():
    print("--- Testing Filepath Auto-Resolution ---")
    base_name = "./test_resolution"
    clean_shards(base_name)
    
    # 1. Write some test shards
    sentences = ["Line 1 of dataset", "Line 2 of dataset"]
    rox = Roxxel(filepath=f"{base_name}.rox")
    rox.write(sentences, block_size=64, max_shard_bytes=80, separator=None)
    
    # Verify shards were created
    shards = sorted(glob.glob(f"{base_name}_*.rox"))
    assert len(shards) > 0
    
    # 2. Test prefix resolution without extension: "./test_resolution"
    rox_prefix = Roxxel(filepath=base_name)
    rox_prefix.open()
    assert len(rox_prefix.filepaths) == len(shards)
    assert rox_prefix.filepaths == shards
    
    # 3. Test prefix resolution with extension: "./test_resolution.rox"
    rox_ext = Roxxel(filepath=f"{base_name}.rox")
    rox_ext.open()
    assert len(rox_ext.filepaths) == len(shards)
    assert rox_ext.filepaths == shards

    # 4. Test directory resolution
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    try:
        for shard in shards:
            shutil.copy(shard, temp_dir)
            
        rox_dir = Roxxel(filepath=temp_dir)
        rox_dir.open()
        assert len(rox_dir.filepaths) == len(shards)
        assert all(os.path.basename(f) in [os.path.basename(s) for s in shards] for f in rox_dir.filepaths)
    finally:
        shutil.rmtree(temp_dir)
        
    clean_shards(base_name)
    print("Filepath Auto-Resolution passed successfully!\n")

if __name__ == "__main__":
    test_fused_sharded_mode()
    test_int32_tokenized_dataset()
    test_logger()
    test_incomplete_shard_recovery()
    test_multiphase_curriculum_stream()
    test_mixer_multiphase_stream()
    test_filepath_auto_resolution()
