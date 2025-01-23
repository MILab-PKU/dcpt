import bisect
import warnings
from typing import Generic, Iterable, List, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`. Subclasses could also
    optionally implement :meth:`__getitems__`, for speedup batched samples
    loading. This method accepts list of indices of samples of batch and returns
    list of samples.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs an index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    # def __getitems__(self, indices: List) -> List[T_co]:
    # Not implemented to prevent false-positives in fetcher check in
    # torch.utils.data._utils.fetch._MapDatasetFetcher

    def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
        return ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py


class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence, enlarge_ratios):
        r, s = [], 0
        for e, ratio in zip(sequence, enlarge_ratios):
            l = len(e) * ratio
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset], enlarge_ratios: List[int]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        self.enlarge_ratios = enlarge_ratios
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        assert len(self.datasets) == len(self.enlarge_ratios), f"The numbers of datasets is not the same as the numbers of enlarge_ratios, {len(datasets)} v.s. {len(enlarge_ratios)}."  # type: ignore[arg-type]
        self.cumulative_sizes = self.cumsum(self.datasets, self.enlarge_ratios)
        self.datasets_length = []
        for d in self.datasets:
            self.datasets_length.append(len(d))

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx % self.datasets_length[dataset_idx]
        else:
            sample_idx = (
                idx - self.cumulative_sizes[dataset_idx - 1]
            ) % self.datasets_length[dataset_idx]
        data = self.datasets[dataset_idx][sample_idx]
        data["dataset_idx"] = dataset_idx
        return data

    @property
    def cummulative_sizes(self):
        warnings.warn(
            "cummulative_sizes attribute is renamed to " "cumulative_sizes",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cumulative_sizes
