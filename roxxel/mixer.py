import numpy as np
import jax
from typing import List, Dict
from roxxel.core import Roxxel, RoxxelStream

class RoxxelMixer:
    """
    Combines N Roxxel datasets into a single JAX stream with custom mixing weights
    while preserving absolute determinism and O(1) resume capabilities.
    """
    def __init__(self, datasets: Dict[str, Roxxel], weights: Dict[str, float]):
        if not datasets:
            raise ValueError("At least one dataset must be provided.")
        if set(datasets.keys()) != set(weights.keys()):
            raise ValueError("Datasets and weights must have matching keys.")
            
        self.datasets = datasets
        self.names = list(datasets.keys())
        
        # Normalize weights to probabilities
        total_w = sum(weights.values())
        if total_w <= 0:
            raise ValueError("Total weights must be positive.")
        self.probs = np.array([weights[name] / total_w for name in self.names], dtype=np.float64)

    def stream(
        self,
        seq_len: int,
        batch_size: int = 32,
        seed: int = 42,
        start_step: int = 0,
        completed_phases: List[tuple] = None,
        total_steps: int = None,
        dtype = np.int32,
        mesh = None,
        data_sharding = None
    ) -> RoxxelStream:
        # Expose class native dtype matching first dataset's configuration
        first_dataset = self.datasets[self.names[0]]
        if not first_dataset._is_open:
            first_dataset.open()
            
        native_dtype_name = getattr(first_dataset, "dtype", "uint8")
        native_dtype = np.dtype(native_dtype_name)
        element_size = native_dtype.itemsize

        # 1. Determine a safe upper bound on total global steps to pre-generate choices
        max_possible_global_steps = 0
        for name in self.names:
            dataset = self.datasets[name]
            if not dataset._is_open:
                dataset.open()
            # Total steps this dataset can support individually at current resolution
            compile_block_size = len(dataset[0])
            total_dataset_bytes = len(dataset) * compile_block_size
            total_bytes_per_batch = batch_size * seq_len * element_size
            max_possible_global_steps += total_dataset_bytes // total_bytes_per_batch

        if completed_phases:
            max_possible_global_steps += sum(p[0] for p in completed_phases)

        # Draw all choices for the entire history and future at once to ensure absolute consistency
        rng_sim = np.random.default_rng(seed)
        all_choices = rng_sim.choice(len(self.names), size=max_possible_global_steps, p=self.probs)

        # 2. Simulate historical steps allocation per dataset
        local_completed_phases = {name: [] for name in self.names}
        global_accumulator = 0
        
        if completed_phases:
            for p_steps, p_batch_size, p_seq_len in completed_phases:
                # Slice subarray of selections made during this historical phase
                choices = all_choices[global_accumulator : global_accumulator + p_steps]
                counts = np.bincount(choices, minlength=len(self.names))
                
                for i, name in enumerate(self.names):
                    local_completed_phases[name].append((counts[i], p_batch_size, p_seq_len))
                global_accumulator += p_steps

        # 3. Simulate current phase allocations up to start_step
        current_phase_steps = start_step - global_accumulator
        local_start_steps = {name: 0 for name in self.names}
        
        if current_phase_steps > 0:
            choices = all_choices[global_accumulator : start_step]
            counts = np.bincount(choices, minlength=len(self.names))
            for i, name in enumerate(self.names):
                local_start_steps[name] = counts[i]

        # 4. Instantiate underlying local streams
        local_streams = {}
        for i, name in enumerate(self.names):
            local_streams[name] = self.datasets[name].stream(
                seq_len=seq_len,
                batch_size=batch_size,
                seed=seed + i + 1,  # Deterministic distinct seeds per dataset
                start_step=local_start_steps[name],
                completed_phases=local_completed_phases[name],
                total_steps=None,  # We let mixer control total steps
                dtype=dtype,
                mesh=mesh,
                data_sharding=data_sharding
            )

        # 5. Determine total remaining steps
        if total_steps is None:
            # Estimate remaining global steps based on local capacity of each dataset
            max_steps = None
            for name in self.names:
                rem_steps = len(local_streams[name])
                prob = self.probs[self.names.index(name)]
                if prob > 0:
                    est = int(rem_steps / prob)
                    if max_steps is None or est < max_steps:
                        max_steps = est
            total_steps = max_steps if max_steps is not None else 0

        # 6. Generator yielding JAX device arrays
        def mix_generator():
            for step_idx in range(total_steps):
                global_step = start_step + step_idx
                if global_step >= len(all_choices):
                    return
                dataset_idx = all_choices[global_step]
                active_name = self.names[dataset_idx]
                
                try:
                    yield next(local_streams[active_name])
                except StopIteration:
                    return

        return RoxxelStream(mix_generator(), total_steps)
