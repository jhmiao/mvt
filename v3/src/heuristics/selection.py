from typing import List
from src.solutions.solutions import Solution


def select_top_k(
    solutions: List[Solution],
    k: int,
) -> List[Solution]:
    """
    Keep the k best solutions by smallest objective value.
    """
    return sorted(
        solutions,
        key=lambda s: s.objective_value
    )[:k]
