# src/models/travel_model.py
from typing import Literal
from typing import TYPE_CHECKING


from .problem_data import ProblemData  # only for type hints; no runtime import
from .solution import Route, Solution

class TravelCost:
    """
    Provides travel times between nodes for given nurses.
    Wraps ProblemData arrays and indexing rules.
    """

    def __init__(self, pd):
        self.pd = pd
        self.m = pd.m
        self.DEPOT_AM = pd.m + 1
        self.DEPOT_PM = pd.m + 2

    def home_index(self, w: int) -> int:
        """Return the unique home index for nurse w."""
        return 100 + w

    def cost(self, w: int, i: int, j: int) -> int:
        """
        Return travel time (minutes) for arc i->j for nurse w.
        """
        pd = self.pd
        m, AM, PM = self.m, self.DEPOT_AM, self.DEPOT_PM
        HOME = self.home_index(w)

        # home -> event
        if i == HOME and 0 <= j < m:
            return int(pd.C_home[w, j])

        # event -> home
        if 0 <= i < m and j == HOME:
            return int(pd.C_home[w, i])

        # event -> event
        if 0 <= i < m and 0 <= j < m:
            return int(pd.C_event[i, j])

        # event -> depot PM
        if 0 <= i < m and j == PM:
            return int(pd.C_depot_e[i])

        # depot AM -> event
        if i == AM and 0 <= j < m:
            return int(pd.C_depot_e[j])

        # home -> depot AM (start-of-day)
        if i == HOME and j == AM:
            return int(pd.C_depot_h[w])

        # depot PM -> home (end-of-day)
        if i == PM and j == HOME:
            return int(pd.C_depot_h[w])

        # fall back (e.g., HOME->HOME or depot->depot): 0
        return 0
    
    def total_cost(self, sol: Solution) -> int:
        """
        Compute total travel cost for every nurse in the given solution.
        """
        total = 0
        for w in range(self.pd.n):
            for route in sol.iter_routes():
                if route.nurse != w:
                    continue
                nodes = route.nodes
                for idx in range(len(nodes) - 1):
                    i = nodes[idx]
                    j = nodes[idx + 1]
                    total += self.cost(w, i, j)
        return total
