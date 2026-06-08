# Curriculum & Dataset Blending

Curriculum learning is an effective technique for training large language models (LLMs) and sequence models (like State Space Models or Transformers). By starting with short sequences and small batch sizes and progressively shifting to longer contexts and larger batch sizes, you optimize computational efficiency and stabilize early training.

Roxxel provides first-class support for multi-phase curriculum schedules and dataset blending natively using `Phase` and `Curriculum` abstractions.

---

## Defining a Training Curriculum

Below is an example of setting up a multi-phase training roadmap. In Phase 1 we run base pre-training with a short sequence length, and in Phase 2 we extend the context window. We also blend our primary dataset with a secondary domain-specific dataset.

```python
from roxxel import Roxxel, Phase, Curriculum

# 1. Initialize our dataset streamers
primary_ds = Roxxel("./dataset_primary_*.rox")
domain_ds = Roxxel("./dataset_domain_*.rox")

# 2. Setup secondary datasets dict
mix_datasets = {
    "domain_specific": domain_ds
}

# 3. Define the curriculum timeline
# Format: Phase(steps, batch_size, seq_len, optional_weights)
phases = [
    # Phase 1: Base pre-training (90% primary, 10% domain-specific)
    Phase(
        steps=5000, 
        batch_size=16, 
        seq_len=1024, 
        weights={"self": 0.9, "domain_specific": 0.1}
    ),
    # Phase 2: Context window extension (50% primary, 50% domain-specific)
    Phase(
        steps=1000, 
        batch_size=2, 
        seq_len=8192, 
        weights={"self": 0.5, "domain_specific": 0.5}
    )
]

# 4. Instantiate the curriculum schedule
curriculum = Curriculum(
    primary_streamer=primary_ds,
    phases=phases,
    mix_streamers=mix_datasets
)
```

---

## Core Features

### 1. Declaring Dynamic Phase Profiles
Each `Phase` instance encapsulates:
- `steps`: Target training steps for this specific profile.
- `batch_size`: The batch size to yield to device accelerators.
- `seq_len`: The sequence length of each sample in the batch.
- `weights`: Blending ratios (summing to 1.0) defining how samples are mixed between datasets.

### 2. Multi-Dataset Blending (Mixing)
By passing secondary Roxxel dataset streamers to `mix_streamers` and assigning a dictionary of weights to a phase, Roxxel handles process-safe dataset blending. The streamer selects the dataset for each global step using a deterministic random choice (seeded unified across JAX processes) matching your weights:
- `"self"` refers to the primary dataset (`primary_streamer`).
- Other keys map directly to the secondary streamers declared in `mix_streamers`.

### 3. Exhaustion Re-normalization
If one of your blended datasets is fully exhausted (runs out of records) in the middle of a phase, Roxxel will automatically capture the exhaustion, remove the empty dataset from the mix, and re-normalize the remaining active datasets' weights. This guarantees that training continues without crashing or stalling.

---

## API Reference

### Phase
::: roxxel.trainer.Phase
    options:
      show_root_heading: true
      heading_level: 3

### Curriculum
::: roxxel.trainer.Curriculum
    options:
      show_root_heading: true
      heading_level: 3
