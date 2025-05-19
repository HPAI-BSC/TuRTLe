from warnings import warn

from datasets import load_dataset
from tasks.base import Task


class TaskExtension(Task):
    """Extiende Task con trust_remote_code=True en la carga del dataset."""
    
    def __init__(self, stop_words=None, requires_execution=True):
        self.stop_words = stop_words
        self.requires_execution = requires_execution
        try:
            # Modificación clave: añadir trust_remote_code=True
            self.dataset = load_dataset(
                path=self.DATASET_PATH,
                name=self.DATASET_NAME,
                trust_remote_code=True  # <-- Cambio del desarrollador
            )
        except Exception as e:
            warn(
                f"Loading the dataset failed with {str(e)}. This task will use a locally downloaded dataset, "
                "not from the HF hub. This is expected behavior for the DS-1000 benchmark but not for other benchmarks!"
            )
