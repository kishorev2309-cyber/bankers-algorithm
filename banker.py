"""Core Banker's Algorithm logic (NumPy only, no UI code)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SafetyResult:
    is_safe: bool
    sequence: List[int]                      # process indices in safe order
    steps: List[dict] = field(default_factory=list)


@dataclass
class RequestResult:
    granted: bool
    message: str
    safety: Optional[SafetyResult] = None


def calculate_need(maximum: np.ndarray, allocation: np.ndarray) -> np.ndarray:
    """Need = Maximum - Allocation."""
    return maximum - allocation


def validate_system(available: np.ndarray, allocation: np.ndarray,
                    maximum: np.ndarray) -> Optional[str]:
    """Return an error message if the input is invalid, else None."""
    if np.any(available < 0) or np.any(allocation < 0) or np.any(maximum < 0):
        return "Values cannot be negative."
    bad = np.argwhere(allocation > maximum)
    if bad.size:
        p, r = bad[0]
        return (f"Allocation exceeds Maximum for process index {p}, "
                f"resource index {r}. Allocation must be <= Maximum.")
    return None


def safety_check(available: np.ndarray, allocation: np.ndarray,
                 need: np.ndarray) -> SafetyResult:
    """Run the safety algorithm and return the safe sequence (if any)."""
    n = allocation.shape[0]
    work = available.astype(int).copy()
    finish = [False] * n
    sequence: List[int] = []
    steps: List[dict] = []

    progressed = True
    while progressed and len(sequence) < n:
        progressed = False
        for i in range(n):
            if not finish[i] and np.all(need[i] <= work):
                before = work.copy()
                work = work + allocation[i]
                finish[i] = True
                sequence.append(i)
                steps.append({"process": i, "work_before": before,
                              "need": need[i].copy(), "work_after": work.copy()})
                progressed = True

    return SafetyResult(is_safe=all(finish), sequence=sequence, steps=steps)


def process_request(pid: int, request: np.ndarray, available: np.ndarray,
                    allocation: np.ndarray, need: np.ndarray
                    ) -> Tuple[RequestResult, np.ndarray, np.ndarray, np.ndarray]:
    """Verify a request and grant it only if the system stays safe.

    Returns (result, available, allocation, need); the arrays are updated
    copies when granted and unchanged copies otherwise.
    """
    available = available.copy()
    allocation = allocation.copy()
    need = need.copy()

    if np.any(request < 0):
        return RequestResult(False, "Request contains negative values."), available, allocation, need
    if not np.any(request > 0):
        return RequestResult(False, "Request is empty (all zeros)."), available, allocation, need
    if np.any(request > need[pid]):
        return RequestResult(
            False, "Denied: process requested more than its remaining Need "
                   "(exceeded its maximum claim)."), available, allocation, need
    if np.any(request > available):
        return RequestResult(
            False, "Denied: requested resources are not currently available "
                   "(process must wait)."), available, allocation, need

    # Pretend to allocate, then test safety.
    t_avail = available - request
    t_alloc = allocation.copy()
    t_need = need.copy()
    t_alloc[pid] += request
    t_need[pid] -= request

    safety = safety_check(t_avail, t_alloc, t_need)
    if not safety.is_safe:
        return RequestResult(
            False, "Denied: granting this request would leave the system in "
                   "an UNSAFE state.", safety), available, allocation, need

    return RequestResult(True, "Granted: the system remains in a SAFE state.",
                         safety), t_avail, t_alloc, t_need
