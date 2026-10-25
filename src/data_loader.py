from datasets import load_dataset, DatasetDict
import pandas as pd

class HFDatasetLoader:
    """Carga un dataset de Hugging Face y lo devuelve como DataFrame de pandas."""

    def __init__(self, dataset_id: str, split: str | None = None):
        if not dataset_id:
            raise ValueError("Debes indicar el identificador del dataset.")
        self.dataset_id = dataset_id
        self.split = split

    def load(self) -> pd.DataFrame:
        if self.split is not None:
            dataset = load_dataset(self.dataset_id, split=self.split)
        else:
            dataset = load_dataset(self.dataset_id)

        if isinstance(dataset, DatasetDict):
            splits_disponibles = list(dataset.keys())
            self.split = "train" if "train" in splits_disponibles else splits_disponibles[0]
            dataset = dataset[self.split]

        return dataset.to_pandas()