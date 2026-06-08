import os
import glob
import struct
import bisect
import queue
import threading
import numpy as np
import jax

class RoxxelStream:
    """
    A wrapper around the Python generator returned by stream() that exposes 
    the exact __len__ of training steps and prefetches batches asynchronously
    on a background thread to eliminate data starvation on accelerators.
    """
    def __init__(self, generator, total_steps, prefetch_size: int = 4):
        """
        Args:
            generator: The underlying generator yielding batches.
            total_steps (int): The total number of steps in this stream.
            prefetch_size (int): The number of batches to prefetch asynchronously.
                Set to 0 to disable prefetching. Defaults to 4.
        """
        self.generator = generator
        self.total_steps = total_steps
        self.prefetch_size = prefetch_size
        self._queue = queue.Queue(maxsize=prefetch_size) if prefetch_size > 0 else None
        self._thread = None
        self._exception = None
        self._started = False
        self._stop_event = threading.Event()

    def _prefetch_worker(self):
        try:
            for item in self.generator:
                if self._stop_event.is_set():
                    break
                
                # Block until we can put the item, checking for cancellation
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                        
            if not self._stop_event.is_set():
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(None, timeout=0.1)  # Sentinel
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            self._exception = e
            while not self._stop_event.is_set():
                try:
                    self._queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def __iter__(self):
        if not self._started and self.prefetch_size > 0:
            self._started = True
            self._thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self._thread.start()
        return self

    def __next__(self):
        if self.prefetch_size > 0:
            if not self._started:
                self.__iter__()
            item = self._queue.get()
            if self._exception is not None:
                raise self._exception
            if item is None:
                raise StopIteration
            return item
        else:
            return next(self.generator)

    def close(self):
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        self._thread = None
            
    def __del__(self):
        self.close()

    def __len__(self):
        return self.total_steps

class Roxxel:
    """
    A bare-bones, zero-RAM sharded block-based dataset manager.
    Packs arbitrary data streams into strictly uniform blocks on disk,
    virtualizes sharded structures, and streams high-performance JAX/NumPy batches.
    """
    MAGIC_SIGNATURE = b"ROXXEL02"  # 8-byte secure signature tag

    def __init__(self, filepath="./stream_reservoir.rox"):
        """
        Args:
            filepath (str or list of str): A single file path, glob pattern, directory,
                or a list of file paths containing Roxxel shards.
        """
        self.raw_data = None
        self.index_table = None
        self._total_records = 0
        self._is_open = False
        self._shards = []
        self._shard_boundaries = []
        self.raw_filepath = None

        # Support single string, list of strings, or glob patterns
        if isinstance(filepath, list):
            self.filepaths = filepath
        elif isinstance(filepath, str):
            self.raw_filepath = filepath
            if "*" in filepath or "?" in filepath:
                self.filepaths = sorted(glob.glob(filepath))
            else:
                self.filepaths = [filepath]
        else:
            raise TypeError("filepath must be a string (file/pattern) or a list of strings.")

    # =====================================================================
    # API 1: FUSED FIXED-BLOCK WRITE STREAM (WITH SHARDING)
    # =====================================================================
    def write(self, data_generator, separator: bytes, block_size: int = 4096, max_shard_bytes: int = None, dtype: str = None):
        """
        Accepts a stream of strings, bytes, or numpy arrays, packs them into strictly uniform
        blocks of `block_size` bytes (with padding), and writes them to shards or a single file.

        Args:
            data_generator: Iterator yielding strings, raw bytes, bytearrays, or numpy arrays.
            separator (bytes): Separator appended after each item in the stream.
            block_size (int, optional): The target size of each uniform block in bytes. Defaults to 4096.
            max_shard_bytes (int, optional): Maximum size of each shard file in bytes. If exceeded,
                a new shard is created. Defaults to None (single file).
            dtype (str, optional): Target numpy dtype for the items. If None, it is automatically detected.
                Defaults to None.
        """
        self.close()

        # Deduce a write path even if self.filepaths is empty due to a new glob pattern
        if len(self.filepaths) == 0:
            if self.raw_filepath:
                base_path = self.raw_filepath
            else:
                raise ValueError("No filepath specified to write to.")
        else:
            base_path = self.filepaths[0]

        detected_dtype = dtype

        # Uniform Block generator packing logic
        def uniform_block_generator():
            nonlocal detected_dtype
            buffer = bytearray()
            for item in data_generator:
                if isinstance(item, str):
                    item_bytes = item.encode("utf-8")
                    if detected_dtype is None:
                        detected_dtype = "uint8"
                elif isinstance(item, bytes) or isinstance(item, bytearray):
                    item_bytes = bytes(item)
                    if detected_dtype is None:
                        detected_dtype = "uint8"
                elif isinstance(item, np.ndarray):
                    item_bytes = item.tobytes()
                    if detected_dtype is None:
                        detected_dtype = str(item.dtype)
                else:
                    raise TypeError("Data generator items must be strings, raw bytes/bytearrays, or numpy arrays.")
                
                buffer.extend(item_bytes)
                if separator:
                    buffer.extend(separator)
                
                while len(buffer) >= block_size:
                    yield bytes(buffer[:block_size])
                    del buffer[:block_size]
            
            # Flush trailing residual blocks with padding
            if len(buffer) > 0:
                pad_len = block_size - len(buffer)
                if separator:
                    pad_bytes = (separator * (pad_len // len(separator) + 1))[:pad_len]
                else:
                    pad_bytes = b"\x00" * pad_len
                
                buffer.extend(pad_bytes)
                yield bytes(buffer)

        # Call underlying write orchestrator
        self._write_orchestrator(uniform_block_generator(), max_shard_bytes, lambda: detected_dtype or "uint8")

    def _write_orchestrator(self, block_stream, max_shard_bytes=None, get_dtype=lambda: "uint8"):
        base_path = self.filepaths[0] if len(self.filepaths) > 0 else self.raw_filepath
        if max_shard_bytes is None:
            if "*" in base_path or "?" in base_path:
                base_path = base_path.replace("*", "base").replace("?", "base")
            self._write_single_file(base_path, block_stream, get_dtype)
            return

        if base_path.endswith(".rox"):
            base_name = base_path[:-4]
        else:
            base_name = base_path

        if "*" in base_name or "?" in base_name:
            base_name = base_name.replace("*", "base").replace("?", "base")

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
                
                if last_shard_size >= 32:
                    with open(current_shard_path, "rb") as f:
                        f.seek(last_shard_size - 32)
                        footer_block = f.read(32)
                        total_records, raw_data_size, dtype_bytes, file_signature = struct.unpack("<qq8s8s", footer_block)

                    if file_signature == b"ROXXEL02":
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
            for block_bytes in block_stream:
                payload_size = len(block_bytes)
                estimated_size = current_offset + payload_size + (len(end_offsets) + 1) * 8 + 32
                if estimated_size > max_shard_bytes and len(end_offsets) > 0:
                    f_out.close()
                    self._finalize_shard(current_shard_path, end_offsets, current_offset, get_dtype())
                    
                    shard_idx += 1
                    current_shard_path = f"{base_name}_{shard_idx:04d}.rox"
                    print(f"📦 Shard limit reached. Creating new shard: {current_shard_path}")
                    
                    end_offsets = []
                    current_offset = 0
                    f_out = open(current_shard_path, "ab")

                f_out.write(block_bytes)
                current_offset += payload_size
                end_offsets.append(current_offset)
        finally:
            f_out.close()

        if len(end_offsets) > 0:
            self._finalize_shard(current_shard_path, end_offsets, current_offset, get_dtype())

    def _write_single_file(self, path, block_stream, get_dtype=lambda: "uint8"):
        end_offsets = []
        raw_data_size = 0

        if os.path.exists(path):
            total_file_bytes = os.path.getsize(path)
            if total_file_bytes >= 32:
                with open(path, "rb") as f:
                    f.seek(total_file_bytes - 32)
                    footer_block = f.read(32)
                    total_records, raw_data_size, dtype_bytes, file_signature = struct.unpack("<qq8s8s", footer_block)

                if file_signature == b"ROXXEL02":
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
            else:
                end_offsets = []
                raw_data_size = 0

        current_offset = raw_data_size
        # Truncate file to 0 if starting fresh or overwriting an invalid archive
        if current_offset == 0 and os.path.exists(path):
            open(path, "wb").close()

        with open(path, "ab") as f:
            for block_bytes in block_stream:
                f.write(block_bytes)
                current_offset += len(block_bytes)
                end_offsets.append(current_offset)

        if len(end_offsets) > 0:
            self._finalize_shard(path, end_offsets, current_offset, get_dtype())

    def _finalize_shard(self, path, end_offsets, raw_data_size, dtype="uint8"):
        total_records = len(end_offsets)
        # Pad or truncate dtype to exactly 8 bytes
        dtype_bytes = dtype.encode("utf-8")
        if len(dtype_bytes) < 8:
            dtype_bytes = dtype_bytes + b"\x00" * (8 - len(dtype_bytes))
        elif len(dtype_bytes) > 8:
            dtype_bytes = dtype_bytes[:8]

        with open(path, "ab") as f:
            np.array(end_offsets, dtype="<i8").tofile(f)
            footer = struct.pack("<qq8s8s", total_records, raw_data_size, dtype_bytes, b"ROXXEL02")
            f.write(footer)
        print(f"✅ Finalized shard {os.path.basename(path)} - Records: {total_records}, Data Bytes: {raw_data_size}, Dtype: {dtype}")

    # =====================================================================
    # API 2: READ / LOAD (SHARDED SEQUENCE INTERFACE)
    # =====================================================================
    def open(self):
        """
        Memory maps all files in the sharded dataset for high-performance read-only access.

        Raises:
            FileNotFoundError: If no matching shard files are found.
            ValueError: If a shard file is corrupted (e.g. invalid signature or size).
        """
        if self._is_open:
            return

        self._shards = []
        self._shard_boundaries = []
        self._total_records = 0

        # In case globs returned nothing
        if len(self.filepaths) == 0:
            raise FileNotFoundError("No matching files found for the specified dataset path/pattern.")

        # Resolve directory or prefix paths dynamically on open
        resolved_paths = []
        for path in self.filepaths:
            if os.path.isdir(path):
                dir_shards = sorted(glob.glob(os.path.join(path, "*.rox")))
                resolved_paths.extend(dir_shards)
            elif not os.path.exists(path):
                prefix = path[:-4] if path.endswith(".rox") else path
                shards = sorted(glob.glob(f"{prefix}_*.rox"))
                if not shards:
                    shards = sorted(glob.glob(f"{prefix}*.rox"))
                if shards:
                    resolved_paths.extend(shards)
                else:
                    resolved_paths.append(path)
            else:
                resolved_paths.append(path)

        if len(resolved_paths) == 0:
            raise FileNotFoundError("No matching files found for the specified dataset path/pattern.")

        self.filepaths = resolved_paths

        for path in resolved_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing dataset shard file at {path}.")

            total_file_bytes = os.path.getsize(path)
            if total_file_bytes < 32:
                raise ValueError(f"Corrupted shard {path}: size is less than 32-byte footer size.")

            with open(path, "rb") as f:
                f.seek(total_file_bytes - 32)
                footer_block = f.read(32)
                total_records, raw_data_size, dtype_bytes, file_signature = struct.unpack("<qq8s8s", footer_block)
                if file_signature != b"ROXXEL02":
                    raise ValueError(f"Corrupted signature in shard {path}.")
                dtype = dtype_bytes.decode("utf-8").strip("\x00")

            # Open standard python file handle for safe, pythonic descriptor management
            f_handle = open(path, "rb")

            # Memory map the raw data and index table using the file handle
            raw_data = np.memmap(
                f_handle,
                dtype=np.uint8,
                mode="r",
                offset=0,
                shape=(raw_data_size,)
            )

            index_table = np.memmap(
                f_handle,
                dtype=np.int64,
                mode="r",
                offset=raw_data_size,
                shape=(total_records,)
            )

            self._shards.append({
                "file_handle": f_handle,
                "raw_data": raw_data,
                "index_table": index_table,
                "total_records": total_records,
                "dtype": dtype
            })

            self._total_records += total_records
            self._shard_boundaries.append(self._total_records)

        # Expose primary shard properties for backward-compatibility if only 1 file exists
        if len(self._shards) == 1:
            self.raw_data = self._shards[0]["raw_data"]
            self.index_table = self._shards[0]["index_table"]
            self.dtype = self._shards[0]["dtype"]
        elif len(self._shards) > 1:
            self.dtype = self._shards[0]["dtype"]

        self._is_open = True

    def close(self):
        """
        Closes all mapped file handles and clears metadata.
        """
        if not self._is_open:
            return

        for shard in self._shards:
            # Delete references to the memmap objects
            del shard["raw_data"]
            del shard["index_table"]
            
            # Cleanly close the underlying Python file handle
            if shard["file_handle"] is not None:
                shard["file_handle"].close()

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

    def estimate_steps(self, seq_len: int, batch_size: int) -> int:
        """
        Calculates the exact number of training steps per epoch for a given sequence length and batch size.

        Args:
            seq_len (int): Sequence length of each training sample.
            batch_size (int): The number of samples per batch.

        Returns:
            The exact number of steps in an epoch.
        """
        if not self._is_open:
            self.open()
        total_blocks = len(self)
        if total_blocks == 0:
            return 0
        
        compile_block_size = len(self[0])
        native_dtype_name = getattr(self, "dtype", "uint8")
        native_dtype = np.dtype(native_dtype_name)
        element_size = native_dtype.itemsize
        
        total_bytes = total_blocks * compile_block_size
        total_bytes_per_batch = batch_size * seq_len * element_size
        return total_bytes // total_bytes_per_batch

    # =====================================================================
    # API 3: UNIFIED SEQUENCE STREAMING ENGINE (NUMPY / JAX)
    # =====================================================================
    def stream(self, seq_len: int, batch_size: int, seed: int, start_step: int = 0, completed_phases: list = None, total_steps: int = None, dtype = np.int32, mesh = None, data_sharding = None, mix_datasets: dict = None, weights: dict = None, prefetch_size: int = 4) -> RoxxelStream:
        """
        Streams from an open Roxxel instance with absolute bit-level determinism.
        Supports multi-phase curriculum training with N phases having different 
        batch sizes and sequence lengths.
        
        Optionally mixes multiple Roxxel datasets according to specified weights.

        Args:
            seq_len (int): Sequence length of each training sample.
            batch_size (int): The number of samples per batch.
            seed (int): Random seed for shuffle reproducibility.
            start_step (int, optional): The global step to resume streaming from. Defaults to 0.
            completed_phases (list of tuple, optional): List of (steps, batch_size, seq_len)
                for historical training phases. Used to calculate skips accurately. Defaults to None.
            total_steps (int, optional): Limit the stream to a maximum number of steps. Defaults to None.
            dtype (numpy dtype, optional): The datatype of the training arrays to yield. Defaults to np.int32.
            mesh (jax.sharding.Mesh, optional): JAX hardware device mesh for sharding. If None,
                it is automatically generated. Defaults to None.
            data_sharding (jax.sharding.NamedSharding, optional): JAX named sharding specification.
                If None, it is automatically generated. Defaults to None.
            mix_datasets (dict, optional): Dict mapping names to other Roxxel dataset instances to mix.
                Defaults to None.
            weights (dict, optional): Dict mapping dataset names to mixing weights. Defaults to None.

        Returns:
            An iterator yielding JAX device arrays.
        """
        if mix_datasets and weights:
            # Check that keys match
            all_datasets = {"self": self}
            all_datasets.update(mix_datasets)
            
            if set(all_datasets.keys()) != set(weights.keys()):
                raise ValueError("mix_datasets/self and weights must have matching keys.")
                
            names = list(all_datasets.keys())
            
            # Determine total bytes for each dataset
            dataset_bytes = []
            for name in names:
                ds = all_datasets[name]
                if not ds._is_open:
                    ds.open()
                compile_block_size = len(ds[0])
                total_bytes = len(ds) * compile_block_size
                dataset_bytes.append(total_bytes)

            # We use a deterministic RNG for the entire simulation
            rng_sim = np.random.default_rng(seed)
            
            # Track remaining bytes for each dataset
            rem_bytes = {name: dataset_bytes[names.index(name)] for name in names}
            
            # We track local steps completed by each dataset in each phase
            local_completed_phases = {name: [] for name in names}
            local_start_steps = {name: 0 for name in names}
            
            # List of historical phases plus the current phase up to start_step
            phases_to_simulate = []
            if completed_phases:
                for p_steps, p_batch_size, p_seq_len in completed_phases:
                    phases_to_simulate.append((p_steps, p_batch_size, p_seq_len, False))
            # Add the current phase up to start_step
            current_phase_steps = start_step - sum(p[0] for p in completed_phases) if completed_phases else start_step
            if current_phase_steps > 0:
                phases_to_simulate.append((current_phase_steps, batch_size, seq_len, True))
                
            # Run the step-by-step simulation
            for p_steps, p_batch_size, p_seq_len, is_current in phases_to_simulate:
                p_bytes_per_step = {}
                for name in names:
                    ds = all_datasets[name]
                    native_dtype_name = getattr(ds, "dtype", "uint8")
                    el_size = np.dtype(native_dtype_name).itemsize
                    p_bytes_per_step[name] = p_batch_size * p_seq_len * el_size
                    
                phase_counts = {name: 0 for name in names}
                
                for t in range(p_steps):
                    active_names = [name for name in names if rem_bytes[name] >= p_bytes_per_step[name]]
                    if not active_names:
                        break
                        
                    if len(active_names) > 1:
                        active_weights = np.array([weights[name] for name in active_names], dtype=np.float64)
                        sum_w = active_weights.sum()
                        if sum_w <= 0:
                            break
                        active_probs = active_weights / sum_w
                        chosen_name = rng_sim.choice(active_names, p=active_probs)
                    else:
                        chosen_name = active_names[0]
                        
                    rem_bytes[chosen_name] -= p_bytes_per_step[chosen_name]
                    phase_counts[chosen_name] += 1
                    
                if is_current:
                    for name in names:
                        local_start_steps[name] = phase_counts[name]
                else:
                    for name in names:
                        local_completed_phases[name].append((phase_counts[name], p_batch_size, p_seq_len))

            # 4. Instantiate underlying streams recursively
            local_streams = {}
            for i, name in enumerate(names):
                local_streams[name] = all_datasets[name].stream(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    seed=seed + i + 1,
                    start_step=local_start_steps[name],
                    completed_phases=local_completed_phases[name],
                    total_steps=None,
                    dtype=dtype,
                    mesh=mesh,
                    data_sharding=data_sharding,
                    mix_datasets=None,
                    weights=None,
                    prefetch_size=0  # Disable prefetching on internal streams to save CPU threads!
                )

            # 5. Determine the remaining steps and the active choices sequence
            cur_bytes_per_step = {}
            for name in names:
                ds = all_datasets[name]
                native_dtype_name = getattr(ds, "dtype", "uint8")
                el_size = np.dtype(native_dtype_name).itemsize
                cur_bytes_per_step[name] = batch_size * seq_len * el_size

            active_choices = []
            step_limit = total_steps if total_steps is not None else float('inf')
            step_idx = 0
            
            while step_idx < step_limit:
                active_names = [name for name in names if rem_bytes[name] >= cur_bytes_per_step[name]]
                if not active_names:
                    break
                    
                if len(active_names) > 1:
                    active_weights = np.array([weights[name] for name in active_names], dtype=np.float64)
                    sum_w = active_weights.sum()
                    if sum_w <= 0:
                        break
                    active_probs = active_weights / sum_w
                    chosen_name = rng_sim.choice(active_names, p=active_probs)
                else:
                    chosen_name = active_names[0]
                    
                rem_bytes[chosen_name] -= cur_bytes_per_step[chosen_name]
                active_choices.append(chosen_name)
                step_idx += 1
                
            total_steps = len(active_choices)

            # 6. Generator
            def mix_generator():
                for chosen_name in active_choices:
                    try:
                        yield next(local_streams[chosen_name])
                    except StopIteration:
                        return

            return RoxxelStream(mix_generator(), total_steps, prefetch_size)

        total_blocks = len(self)
        if total_blocks == 0:
            raise ValueError("Roxxel database is empty or not opened.")
        
        global_indices = np.arange(total_blocks)
        rng = np.random.default_rng(seed)
        rng.shuffle(global_indices)
        
        # Measure compiled block size from the first record
        compile_block_size = len(self[0]) 
        
        # Read the file's native data type from metadata (defaults to uint8 for older ROXXEL01 files)
        native_dtype_name = getattr(self, "dtype", "uint8")
        native_dtype = np.dtype(native_dtype_name)
        element_size = native_dtype.itemsize
        
        # Calculate bytes per batch for the CURRENT phase layout configuration
        total_bytes_per_batch = batch_size * seq_len * element_size
        dtype = np.dtype(dtype)
        
        # --- GENERALIZED N-PHASE ASYMMETRIC RESUME MATH ---
        total_bytes_to_skip = 0
        current_step_accumulator = 0
        phase_found = False
        
        if completed_phases:
            for p_steps, p_batch_size, p_seq_len in completed_phases:
                if start_step < current_step_accumulator + p_steps:
                    # The start_step resides within this historical phase!
                    steps_in_this_phase = start_step - current_step_accumulator
                    total_bytes_to_skip += steps_in_this_phase * p_batch_size * p_seq_len * element_size
                    current_step_accumulator = start_step
                    phase_found = True
                    break
                else:
                    # This historical phase was fully completed
                    total_bytes_to_skip += p_steps * p_batch_size * p_seq_len * element_size
                    current_step_accumulator += p_steps
        
        if not phase_found:
            # The start_step resides in the current phase
            steps_in_current_phase = start_step - current_step_accumulator
            total_bytes_to_skip += steps_in_current_phase * total_bytes_per_batch
            
        consumed_blocks, remainder_bytes = divmod(total_bytes_to_skip, compile_block_size)
        
        if consumed_blocks > 0:
            global_indices = global_indices[consumed_blocks:]
            print(f"⏭️ Roxxel instantly jumped past {consumed_blocks} blocks. Resuming at global step {start_step}.")
        
        # Determine the total remaining steps for this stream session context window
        total_dataset_bytes = total_blocks * compile_block_size
        remaining_bytes = max(0, total_dataset_bytes - total_bytes_to_skip)
        max_possible_steps = remaining_bytes // total_bytes_per_batch
        
        if total_steps is None:
            total_steps = max_possible_steps
        else:
            total_steps = min(total_steps, max_possible_steps)

        # Always construct and use JAX data sharding for device array streaming
        if mesh is None or data_sharding is None:
            from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
            from jax.experimental import mesh_utils
            devices = jax.devices()
            mesh = Mesh(mesh_utils.create_device_mesh((len(devices),)), axis_names=('data',))
            data_sharding = NamedSharding(mesh, P('data', None))

        def batch_generator():
            record_ptr = 0
            leftover_chunk = None
            
            # Pre-fill leftover chunk
            if remainder_bytes > 0 and record_ptr < len(global_indices):
                idx = global_indices[record_ptr]
                raw_slice = self[int(idx)]
                leftover_chunk = raw_slice[remainder_bytes:]
                record_ptr += 1
            
            for _ in range(total_steps):
                chunks = []
                collected_bytes = 0
                
                if leftover_chunk is not None:
                    chunks.append(leftover_chunk)
                    collected_bytes += len(leftover_chunk)
                    leftover_chunk = None
                    
                while collected_bytes < total_bytes_per_batch:
                    if record_ptr >= len(global_indices):
                        return
                    idx = global_indices[record_ptr]
                    raw_slice = self[int(idx)]
                    chunks.append(raw_slice)
                    collected_bytes += len(raw_slice)
                    record_ptr += 1
                    
                if len(chunks) == 1:
                    full_chunk = chunks[0]
                else:
                    full_chunk = np.concatenate(chunks)
                
                if collected_bytes > total_bytes_per_batch:
                    leftover_chunk = full_chunk[total_bytes_per_batch:]
                    full_chunk = full_chunk[:total_bytes_per_batch]
                
                # full_chunk is guaranteed to be contiguous uint8 array here
                # Parse using the dataset's native dtype, then cast to target training dtype
                # Using numpy views to avoid multiple expensive python string/byte copies
                try:
                    flat_tokens = full_chunk.view(native_dtype).astype(dtype)
                except ValueError:
                    # Fallback if alignment somehow prevents .view()
                    flat_tokens = np.frombuffer(full_chunk.tobytes(), dtype=native_dtype).astype(dtype)
                    
                numpy_batch = flat_tokens.reshape(batch_size, seq_len)
                
                # Yield high-performance JAX device array
                yield jax.device_put(numpy_batch, data_sharding)

        return RoxxelStream(batch_generator(), total_steps, prefetch_size)
