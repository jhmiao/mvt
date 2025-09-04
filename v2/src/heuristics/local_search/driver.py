# src/heuristics/local_search/driver.py
from .operators.two_opt import best_two_opt
# from .operators.relocate import best_relocate
# from .operators.swap import best_swap

def improve(solution, ctx, strategy="first_improvement", max_loops=50):
    """Iteratively apply neighborhood moves until local optimum."""
    loops = 0
    improved = True
    while improved and loops < max_loops:
        improved = False
        loops += 1

        # Intra-route 2-opt
        for r_id, route in enumerate(solution.routes):
            cand, delta = best_two_opt(route, ctx.dist, ctx.feasible_check_route)
            if cand is not None:
                solution.apply_route_update(r_id, cand, delta)
                improved = True
                if strategy == "first_improvement": break
        if improved and strategy == "first_improvement": continue

        # # Inter-route moves (relocate/swap)
        # moved = best_relocate(solution, ctx, strategy) or best_swap(solution, ctx, strategy)
        # if moved:
        #     improved = True
    return solution
