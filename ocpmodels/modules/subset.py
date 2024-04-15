from collections.abc import Sequence, Sized
from typing import Any, Generic, Optional, Protocol, runtime_checkable

from torch.utils.data import Dataset
from torch.utils.data import Subset as _Subset
from typing_extensions import TypeVar, override


@runtime_checkable
class CloseableDataset(Protocol):
    def close_db(self) -> None: ...


T = TypeVar("T", infer_variance=True)


class _DatasetMixin(Generic[T]):
    dataset: Dataset[T]

    def close_db(self):
        if not isinstance(self.dataset, CloseableDataset):
            raise NotImplementedError("Dataset does not implement CloseableDataset")
        self.dataset.close_db()


class Subset(_Subset[T], _DatasetMixin[T], Generic[T]):
    pass


class OverfitSubset(Dataset[T], _DatasetMixin[T], Generic[T]):
    def __init__(
        self,
        dataset: Dataset[T],
        samples: int,
        size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.samples = samples
        self.size = size

    def __len__(self):
        if self.size is not None:
            return self.size
        return self.samples

    @override
    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError("Index out of bounds")

        idx = idx % self.samples
        return self.dataset[idx]


def wrap_dataset(
    config: dict[str, Any],
    dataset: Dataset[T],
    split: str,
    *,
    train_dataset: Dataset[T],
):
    # If config['task']['overfit_samples'] is set, only use the first 'overfit_samples' samples
    # of the dataset for training.
    if (overfit_config := config["task"].get("overfit", None)) is not None:
        overfit_samples = overfit_config.get("samples")
        assert (
            isinstance(overfit_samples, int) and overfit_samples > 0
        ), "overfit_samples should be a positive integer"

        overfit_size = overfit_config.get("size", None)

        assert isinstance(train_dataset, Sized), "Dataset should be a Sized object"
        return OverfitSubset(train_dataset, overfit_samples, overfit_size)

    if (subset_config := config["task"].get(f"{split}_subset", None)) is None:
        return dataset

    assert isinstance(dataset, Sized), "Dataset should be a Sized object"

    if isinstance(subset_config, Sequence):
        assert len(subset_config) == 2, "subset_config should be a tuple"
        start, end = subset_config
        start = max(0, start)
        end = min(end, len(dataset))
        assert start < end, "subset_config start should be less than end"
        indices = range(start, end)
    elif isinstance(subset_config, int):
        indices = range(min(subset_config, len(dataset)))
    else:
        raise ValueError("subset_config should be a tuple or an int")
    return Subset(dataset, indices)
