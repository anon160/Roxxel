from roxxel.core import Roxxel
from roxxel.logging import Logger

try:
    from roxxel.trainer import Phase, Curriculum, Trainer, ModelState
    HAS_TRAINER = True
except ImportError:
    HAS_TRAINER = False

if HAS_TRAINER:
    __all__ = ["Roxxel", "Logger", "Phase", "Curriculum", "Trainer", "ModelState"]
else:
    __all__ = ["Roxxel", "Logger"]
