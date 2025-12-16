from src.solutions.solutions import Solution
from src.structures.problem_data import ProblemData
from src.heuristics.config import HeuristicConfig


def improve_solution(
    solution: Solution,
    problem: ProblemData,
    config: HeuristicConfig,
) -> Solution:
    """
    Try to improve a single solution.
    Returns a (possibly) improved solution.
    """

    # placeholder: no-op improvement
    return solution
