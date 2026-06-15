"""Abstract sequence-loader interface.

Every concrete adapter (dummy, KITTI odometry, KITTI-360, Waymo, ...)
implements this protocol so the rest of the pipeline never depends on
the dataset layout on disk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping

from semantic_gs.data.frame import Frame


class SequenceLoader(ABC):
    """One driving sequence accessed as a list of :class:`Frame` objects."""

    # -- metadata --------------------------------------------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """A short human-readable identifier (e.g. ``"kitti_odom/00"``)."""

    @property
    @abstractmethod
    def id2label(self) -> Mapping[int, str]:
        """Class-ID → class-name map matching every ``Frame.label_ids``."""

    # -- frame access ----------------------------------------------------
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Frame: ...

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self[i]

