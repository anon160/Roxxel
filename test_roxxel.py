import os
import glob
import numpy as np
from roxxel import Roxxel

def clean_shards(base_name="test_sharded"):
    for file in glob.glob(f"{base_name}*"):
        os.remove(file)

def test_sharded_write_and_read():
    print("--- Testing Roxxel Sharded Mode ---")
    base_name = "./test_sharded"
    clean_shards(base_name)

    # 1. Generate 20 records of variable sizes
    record_sizes = [5, 10, 15, 20] * 5  # Total 20 records
    original_records = [bytes([i] * size) for i, size in enumerate(record_sizes)]

    # 2. Write with a small shard size (e.g. 50 bytes of raw payload + metadata overhead per shard)
    # This should trigger the creation of multiple shards automatically.
    rox_writer = Roxxel(filepath=f"{base_name}.rox")
    
    # We set max_shard_bytes to 180 to trigger sharding
    rox_writer.write(original_records, max_shard_bytes=180)

    # Verify shards were created on-disk
    created_shards = sorted(glob.glob(f"{base_name}_*.rox"))
    print(f"Created shards: {created_shards}")
    assert len(created_shards) > 1

    # 3. Read using glob pattern
    dataset = Roxxel(filepath=f"{base_name}_*.rox")
    with dataset:
        print(f"Virtualized Sharded Dataset Length: {len(dataset)}")
        assert len(dataset) == len(original_records)

        # Test random access across boundaries
        for i in range(len(dataset)):
            record = dataset[i]
            print(f"  Global Record {i} - Shard-resolved Size: {len(record)}, Unique Value: {record[0]}")
            assert len(record) == record_sizes[i]
            assert np.all(record == i)

        # Test negative indexing
        assert np.all(dataset[-1] == 19)

        # Test slicing
        sliced = dataset[2:7]
        print(f"Sliced [2:7] returns {len(sliced)} items")
        assert len(sliced) == 5
        assert len(sliced[0]) == record_sizes[2]

    # Clean up files
    clean_shards(base_name)
    print("Roxxel Sharded Mode passed successfully!\n")

if __name__ == "__main__":
    test_sharded_write_and_read()
