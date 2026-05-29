import os
import glob
import struct
import bisect
import numpy as np

class Roxxel:
    """
    A bare-bones, zero-RAM single-file or multi-sharded dataset manager.
    Stores raw contiguous payload data, a trailing index table, and a 24-byte footer.
    Seamlessly virtualizes multiple shards on-disk into a single continuous sequence.
    """
    MAGIC_SIGNATURE = b"ROXXEL01"  # 8-byte secure signature tag

    def __init__(self, filepath="./stream_reservoir.rox"):
        self.raw_data = None
        self.index_table = None
        self._total_records = 0
        self._is_open = False
        self._shards = []
        self._shard_boundaries = []

        # Support single string, list of strings, or glob patterns
        if isinstance(filepath, list):
            self.filepaths = filepath
        elif isinstance(filepath, str):
            if "*" in filepath or "?" in filepath:
                self.filepaths = sorted(glob.glob(filepath))
            else:
                self.filepaths = [filepath]
        else:
            raise TypeError("filepath must be a string (file/pattern) or a list of strings.")

    # =====================================================================
    # API 1: WRITE STREAM (WITH SHARDING SUPPORT)
    # =====================================================================
    def write(self, data_generator, max_shard_bytes=None):
        """
        Accepts an iterable stream of raw python byte objects.
        If max_shard_bytes is None, writes/appends to a single file.
        If max_shard_bytes is provided, splits the stream across multiple shards (e.g., dataset_0000.rox).
        """
        self.close()

        if len(self.filepaths) == 0:
            raise ValueError("No filepath specified to write to.")

        if max_shard_bytes is None:
            self._write_single_file(self.filepaths[0], data_generator)
            return

        base_path = self.filepaths[0]
        if base_path.endswith(".rox"):
            base_name = base_path[:-4]
        else:
            base_name = base_path

        # Find first unused shard index
        shard_idx = 0
        while os.path.exists(f"{base_name}_{shard_idx:04d}.rox"):
            shard_idx += 1

        current_shard_path = None
        end_offsets = []
        raw_data_size = 0

        # Try to append to the last existing shard if it has room
        if shard_idx > 0:
            last_shard_path = f"{base_name}_{shard_idx-1:04d}.rox"
            last_shard_size = os.path.getsize(last_shard_path)
            if last_shard_size < max_shard_bytes:
                current_shard_path = last_shard_path
                shard_idx -= 1
                
                with open(current_shard_path, "rb") as f:
                    f.seek(last_shard_size - 24)
                    footer_block = f.read(24)
                    total_records, raw_data_size, file_signature = struct.unpack("<qq8s", footer_block)

                if file_signature == self.MAGIC_SIGNATURE:
                    with open(current_shard_path, "rb") as f:
                        f.seek(raw_data_size)
                        end_offsets = np.fromfile(f, dtype="<i8", count=total_records).tolist()
                    
                    with open(current_shard_path, "r+b") as f:
                        f.truncate(raw_data_size)
                else:
                    current_shard_path = f"{base_name}_{shard_idx:04d}.rox"
                    end_offsets = []
                    raw_data_size = 0
            else:
                current_shard_path = f"{base_name}_{shard_idx:04d}.rox"
        else:
            current_shard_path = f"{base_name}_{shard_idx:04d}.rox"

        current_offset = raw_data_size
        # Truncate file to 0 if starting a fresh or overwritten shard
        if current_offset == 0 and os.path.exists(current_shard_path):
            open(current_shard_path, "wb").close()
            
        f_out = open(current_shard_path, "ab")

        try:
            for item_bytes in data_generator:
                if not isinstance(item_bytes, bytes):
                    raise TypeError("Data generator must exclusively yield raw python 'bytes' objects.")
                
                payload_size = len(item_bytes)
                if payload_size == 0:
                    continue

                # Predict shard size: raw data + index table (8 bytes per record) + 24-byte footer
                estimated_size = current_offset + payload_size + (len(end_offsets) + 1) * 8 + 24
                if estimated_size > max_shard_bytes and len(end_offsets) > 0:
                    f_out.close()
                    self._finalize_shard(current_shard_path, end_offsets, current_offset)
                    
                    shard_idx += 1
                    current_shard_path = f"{base_name}_{shard_idx:04d}.rox"
                    print(f"📦 Shard limit reached. Creating new shard: {current_shard_path}")
                    
                    end_offsets = []
                    current_offset = 0
                    f_out = open(current_shard_path, "ab")

                f_out.write(item_bytes)
                current_offset += payload_size
                end_offsets.append(current_offset)
        finally:
            f_out.close()

        if len(end_offsets) > 0:
            self._finalize_shard(current_shard_path, end_offsets, current_offset)

    def _write_single_file(self, path, data_generator):
        end_offsets = []
        raw_data_size = 0

        if os.path.exists(path):
            total_file_bytes = os.path.getsize(path)
            if total_file_bytes >= 24:
                with open(path, "rb") as f:
                    f.seek(total_file_bytes - 24)
                    footer_block = f.read(24)
                    total_records, raw_data_size, file_signature = struct.unpack("<qq8s", footer_block)

                if file_signature == self.MAGIC_SIGNATURE:
                    print(f"♻️ Found existing archive. Stripping index and footer...")
                    with open(path, "rb") as f:
                        f.seek(raw_data_size)
                        end_offsets = np.fromfile(f, dtype="<i8", count=total_records).tolist()
                    
                    with open(path, "r+b") as f:
                        f.truncate(raw_data_size)
                else:
                    print("⚠️ Invalid signature in existing archive. Overwriting/starting fresh...")
                    end_offsets = []
                    raw_data_size = 0

        current_offset = raw_data_size
        # Truncate file to 0 if starting fresh or overwriting an invalid archive
        if current_offset == 0 and os.path.exists(path):
            open(path, "wb").close()
            
        with open(path, "ab") as f:
            for item_bytes in data_generator:
                if not isinstance(item_bytes, bytes):
                    raise TypeError("Data generator must exclusively yield raw python 'bytes' objects.")
                
                payload_size = len(item_bytes)
                if payload_size == 0:
                    continue

                f.write(item_bytes)
                current_offset += payload_size
                end_offsets.append(current_offset)

        if len(end_offsets) > 0:
            self._finalize_shard(path, end_offsets, current_offset)

    def _finalize_shard(self, path, end_offsets, raw_data_size):
        total_records = len(end_offsets)
        with open(path, "ab") as f:
            np.array(end_offsets, dtype="<i8").tofile(f)
            footer = struct.pack("<qq8s", total_records, raw_data_size, self.MAGIC_SIGNATURE)
            f.write(footer)
        print(f"✅ Finalized shard {os.path.basename(path)} - Records: {total_records}, Data Bytes: {raw_data_size}")

    # =====================================================================
    # API 2: READ / LOAD (SHARDED SEQUENCE INTERFACE)
    # =====================================================================
    def open(self):
        """
        Memory maps all files in the sharded dataset for high-performance read-only access.
        """
        if self._is_open:
            return

        self._shards = []
        self._shard_boundaries = []
        self._total_records = 0

        # In case globs returned nothing
        if len(self.filepaths) == 0:
            raise FileNotFoundError("No matching files found for the specified dataset path/pattern.")

        for path in self.filepaths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing dataset shard file at {path}.")

            total_file_bytes = os.path.getsize(path)
            if total_file_bytes < 24:
                raise ValueError(f"Corrupted shard {path}: size is less than footer size.")

            with open(path, "rb") as f:
                f.seek(total_file_bytes - 24)
                footer_block = f.read(24)
                total_records, raw_data_size, file_signature = struct.unpack("<qq8s", footer_block)

            if file_signature != self.MAGIC_SIGNATURE:
                raise ValueError(f"Corrupted signature in shard {path}.")

            # Memory map the raw data and index table for this shard
            raw_data = np.memmap(
                path,
                dtype=np.uint8,
                mode="r",
                offset=0,
                shape=(raw_data_size,)
            )

            index_table = np.memmap(
                path,
                dtype=np.int64,
                mode="r",
                offset=raw_data_size,
                shape=(total_records,)
            )

            self._shards.append({
                "raw_data": raw_data,
                "index_table": index_table,
                "total_records": total_records
            })

            self._total_records += total_records
            self._shard_boundaries.append(self._total_records)

        # Expose primary shard properties for backward-compatibility if only 1 file exists
        if len(self._shards) == 1:
            self.raw_data = self._shards[0]["raw_data"]
            self.index_table = self._shards[0]["index_table"]

        self._is_open = True

    def close(self):
        """
        Closes all mapped file handles and clears metadata.
        """
        if not self._is_open:
            return

        for shard in self._shards:
            if shard["raw_data"] is not None:
                shard["raw_data"]._mmap.close()
            if shard["index_table"] is not None:
                shard["index_table"]._mmap.close()

        self._shards = []
        self._shard_boundaries = []
        self._total_records = 0
        self.raw_data = None
        self.index_table = None
        self._is_open = False

    def __len__(self):
        if not self._is_open:
            self.open()
        return self._total_records

    def __getitem__(self, idx):
        if not self._is_open:
            self.open()

        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._total_records)
            return [self._get_single_item(i) for i in range(start, stop, step)]
        
        if idx < 0:
            idx += self._total_records

        if idx < 0 or idx >= self._total_records:
            raise IndexError("Record index out of range.")

        return self._get_single_item(idx)

    def _get_single_item(self, idx):
        # Find which shard holds this global index using binary search
        shard_idx = bisect.bisect_right(self._shard_boundaries, idx)
        
        # Calculate local index within that shard
        local_offset = 0 if shard_idx == 0 else self._shard_boundaries[shard_idx - 1]
        local_idx = idx - local_offset
        
        shard = self._shards[shard_idx]
        start = 0 if local_idx == 0 else shard["index_table"][local_idx - 1]
        end = shard["index_table"][local_idx]
        return shard["raw_data"][start:end]

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
