### New module test class

from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np

try:
    from Cubit import Cubit  
except Exception as exc:
    raise ImportError(
    ) from exc


class QCopy:
    def __init__(
        self,
        dimension: int = 2,
        logical_zero_idx: int = 0,
        logical_one_idx: int = 1,
        name: str = "QCopy",
    ) -> None:
        self.original: Cubit = Cubit(
            dimension=dimension,
            logical_zero_idx=logical_zero_idx,
            logical_one_idx=logical_one_idx,
            name=name,
        )
        self.copy: Optional[Cubit] = None
        self.name = name
    def _create_copy_qubit(self) -> Cubit:
        return Cubit(
            dimension=self.original.dimension,
            logical_zero_idx=self.original.logical_zero_idx,
            logical_one_idx=self.original.logical_one_idx,
            name=f"{self.name}_copy",
        )

    def _ensure_copy_exists(self) -> None:
        if self.copy is None:
            self.copy = self._create_copy_qubit()
    def copy_state(self) -> None:
        self._ensure_copy_exists()
        if self.copy is None:
            raise RuntimeError("Failed to create copy qubit.")
        self.copy.initialize(self.original.state.copy())

    def discard_copy(self) -> None:
        self.copy = None

    def has_copy(self) -> bool:
        return self.copy is not None

    def measure_copy(self, logical: bool = True) -> int:
        if self.copy is None:
            raise RuntimeError(
            )
        if logical:
            return self.copy.measure_logical()
        else:
            return self.copy.measure_full()

    def measure_original(self, logical: bool = True) -> int:
        if logical:
            return self.original.measure_logical()
        else:
            return self.original.measure_full()

    def measure_original_and_restore(self, logical: bool = True) -> int:
        self.copy_state()
        if logical:
            outcome = self.original.measure_logical()
        else:
            outcome = self.original.measure_full()
        if self.copy is None:
            raise RuntimeError("Unexpectedly lost the copy qubit during measurement.")
        self.original.initialize(self.copy.state.copy())
        return outcome

    def measure_original_via_copy(self, logical: bool = True) -> int:
        saved_copy = self.copy
        try:
            self.copy = self._create_copy_qubit()
            self.copy.initialize(self.original.state.copy())
            outcome = self.copy.measure_logical() if logical else self.copy.measure_full()
        finally:
            self.copy = saved_copy
        return outcome

    def reset_original(self) -> None:
        self.original.reset_to_logical_zero()

    def reset_copy(self) -> None:
        self._ensure_copy_exists()
        self.copy.reset_to_logical_zero()  

    def initialize_original(self, state: List[complex] | np.ndarray) -> None:
        self.original.initialize(state)
        self.discard_copy()

    def get_logical_populations(self) -> Tuple[float, float]:
        return self.original.get_logical_populations()

    def get_hidden_population(self) -> float:
        return self.original.get_hidden_population_sum()

    def get_statevectors(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        original_sv = self.original.state.copy()
        copy_sv = self.copy.state.copy() if self.copy is not None else None
        return original_sv, copy_sv

    def __repr__(self) -> str:
        status = "has copy" if self.copy is not None else "no copy"
        return f"<QCopy original={self.original.name} ({status})>"
