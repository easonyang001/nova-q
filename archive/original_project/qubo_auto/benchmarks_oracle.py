"""Per-benchmark instance generators and oracle checkers for verification.

Oracle signature: oracle_checker(x: np.ndarray, instance: dict) -> float
  - Returns the problem objective value for binary vector x (lower = better, QUBO minimizes)
  - Returns float('inf') if x is infeasible (violated hard constraint)
  - The test runner brute-forces all 2^n strings independently through both the oracle
    and the QUBO, then checks that their argmin sets overlap.
"""
from __future__ import annotations
from itertools import permutations

import numpy as np


# ---------------------------------------------------------------------------
# Max-Cut
# ---------------------------------------------------------------------------

def max_cut_generator(seed: int, n_variables: int) -> dict:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 9))  # n in [5, 8]; 2^8=256 strings, fast brute force
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.6:
                w = int(rng.integers(1, 6))
                edges.append((i, j, w))
    if not edges:
        edges = [(0, 1, 1)]
    return {"n_nodes": n, "edges": edges}


def max_cut_oracle(x: np.ndarray, instance: dict) -> float:
    """Returns -(cut weight) — negated because QUBO minimizes."""
    edges = instance["edges"]
    cut = sum(w for i, j, w in edges if x[i] != x[j])
    return -float(cut)


# ---------------------------------------------------------------------------
# Number Partition
# ---------------------------------------------------------------------------

def number_partition_generator(seed: int, n_variables: int) -> dict:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 11))  # n in [5, 10]; 2^10=1024 strings, fast brute force
    numbers = [int(rng.integers(1, 20)) for _ in range(n)]
    return {"numbers": numbers, "N": n, "A": sum(numbers)}


def number_partition_oracle(x: np.ndarray, instance: dict) -> float:
    """Returns (sum_subset1 - sum_subset2)^2 — zero iff perfect partition."""
    numbers = instance["numbers"]
    S = sum(numbers)
    sum_B = sum(numbers[i] for i in range(len(numbers)) if x[i] > 0.5)
    return float((S - 2 * sum_B) ** 2)


# ---------------------------------------------------------------------------
# Knapsack
# ---------------------------------------------------------------------------

def knapsack_generator(seed: int, n_variables: int) -> dict:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 9))  # n in [5, 8]; 2^8=256 strings, fast brute force
    weights = [int(rng.integers(1, 8)) for _ in range(n)]
    values = [int(rng.integers(1, 10)) for _ in range(n)]
    capacity = max(1, int(sum(weights) * 0.5))
    return {"n_items": n, "weights": weights, "values": values, "capacity": capacity}


def knapsack_oracle(x: np.ndarray, instance: dict) -> float:
    """Returns -(total value) if feasible (under capacity), else +inf."""
    weights = instance["weights"]
    values = instance["values"]
    capacity = instance["capacity"]
    n = instance["n_items"]
    total_weight = sum(weights[i] for i in range(n) if x[i] > 0.5)
    if total_weight > capacity:
        return float("inf")
    total_value = sum(values[i] for i in range(n) if x[i] > 0.5)
    return -float(total_value)  # negate: QUBO minimizes, knapsack maximizes value


# ---------------------------------------------------------------------------
# Set Cover
# ---------------------------------------------------------------------------

def set_cover_generator(seed: int, n_variables: int) -> dict:
    rng = np.random.default_rng(seed)
    n_elements = int(rng.integers(4, 7))  # 4-6 elements
    n_sets = int(rng.integers(n_elements + 1, n_elements + 4))  # 5-9 sets total

    subsets: list[list[int]] = []
    for _ in range(n_sets):
        size = int(rng.integers(1, min(4, n_elements) + 1))
        subset = sorted(rng.choice(n_elements, size=size, replace=False).tolist())
        subsets.append(subset)

    # Ensure every element appears in at least one subset so the instance is coverable.
    for element in range(n_elements):
        if not any(element in subset for subset in subsets):
            subsets[int(rng.integers(0, n_sets))].append(element)

    subsets = [sorted(set(subset)) for subset in subsets]
    costs = [int(rng.integers(1, 7)) for _ in range(n_sets)]
    return {"n_elements": n_elements, "subsets": subsets, "costs": costs}


def set_cover_oracle(x: np.ndarray, instance: dict) -> float:
    """Returns total selected cost if all elements are covered, else +inf."""
    n_elements = instance["n_elements"]
    subsets = instance["subsets"]
    costs = instance["costs"]
    required = set(range(n_elements))

    covered: set[int] = set()
    total_cost = 0
    for i, selected in enumerate(x):
        if selected > 0.5:
            covered.update(subsets[i])
            total_cost += costs[i]

    if covered != required:
        return float("inf")
    return float(total_cost)


# ---------------------------------------------------------------------------
# TSP (small)
# ---------------------------------------------------------------------------

def tsp_generator(seed: int, n_variables: int) -> dict:
    rng = np.random.default_rng(seed)
    n_cities = 3  # small: n_variables = n^2 = 9
    coords = rng.random((n_cities, 2))
    dist = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            dist[i, j] = np.linalg.norm(coords[i] - coords[j])
    return {"n_cities": n_cities, "distance_matrix": dist.tolist()}


def tsp_oracle(x: np.ndarray, instance: dict) -> float:
    """Returns tour length if x encodes a valid permutation, else +inf."""
    n = instance["n_cities"]
    D = np.array(instance["distance_matrix"])
    assignment = np.array(x).reshape(n, n)

    city_at: dict[int, int] = {}
    for p in range(n):
        col = assignment[:, p]
        assigned = np.where(col > 0.5)[0]
        if len(assigned) != 1:
            return float("inf")
        city_at[p] = int(assigned[0])

    if len(set(city_at.values())) != n:
        return float("inf")

    return float(sum(D[city_at[p], city_at[(p + 1) % n]] for p in range(n)))


# ---------------------------------------------------------------------------
# Graph Coloring
# ---------------------------------------------------------------------------

def graph_coloring_generator(seed: int, n_variables: int) -> dict:
    rng = np.random.default_rng(seed)
    n_nodes = 4
    k_colors = 3
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < 0.5:
                edges.append((i, j))
    return {"n_nodes": n_nodes, "k_colors": k_colors, "edges": edges}


def graph_coloring_oracle(x: np.ndarray, instance: dict) -> float:
    """Returns conflict count if each node has exactly one color, else +inf."""
    n = instance["n_nodes"]
    k = instance["k_colors"]
    edges = instance["edges"]

    assignment = np.array(x).reshape(n, k)
    colors: dict[int, int] = {}
    for i in range(n):
        row = assignment[i]
        c = np.where(row > 0.5)[0]
        if len(c) != 1:
            return float("inf")
        colors[i] = int(c[0])

    conflicts = sum(1 for (i, j) in edges if colors[i] == colors[j])
    return float(conflicts)


# ---------------------------------------------------------------------------
# Maximum Independent Set
# ---------------------------------------------------------------------------

def mis_generator(seed: int, n_variables: int) -> dict:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 9))  # n in [5, 8]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                edges.append((i, j))
    return {"n_nodes": n, "edges": edges}


def mis_oracle(x: np.ndarray, instance: dict) -> float:
    """Returns -(number of selected nodes) if independent set, else +inf."""
    edges = instance["edges"]
    for i, j in edges:
        if x[i] > 0.5 and x[j] > 0.5:
            return float("inf")
    return -float(sum(x > 0.5))


# ---------------------------------------------------------------------------
# Weighted Vertex Cover
# ---------------------------------------------------------------------------

def vertex_cover_generator(seed: int, n_variables: int) -> dict:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 9))  # n in [5, 8]
    weights = [int(rng.integers(1, 10)) for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                edges.append((i, j))
    if not edges:
        edges = [(0, 1)]
    return {"n_nodes": n, "weights": weights, "edges": edges}


def vertex_cover_oracle(x: np.ndarray, instance: dict) -> float:
    """Returns total weight of selected nodes if every edge is covered, else +inf."""
    edges = instance["edges"]
    weights = instance["weights"]
    for i, j in edges:
        if x[i] < 0.5 and x[j] < 0.5:
            return float("inf")
    return float(sum(weights[i] for i in range(len(weights)) if x[i] > 0.5))


# ---------------------------------------------------------------------------
# Portfolio Optimization
# ---------------------------------------------------------------------------

def portfolio_generator(seed: int, n_variables: int) -> dict:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 11))  # n in [6, 10]
    returns = [round(float(rng.uniform(0.02, 0.20)), 3) for _ in range(n)]
    risks = [round(float(rng.uniform(0.01, 0.10)), 3) for _ in range(n)]
    budget = max(2, n // 3)
    return {"n_assets": n, "returns": returns, "risks": risks, "budget": budget}


def portfolio_oracle(x: np.ndarray, instance: dict) -> float:
    """Returns -(return - risk) if exactly budget assets selected, else +inf."""
    returns = instance["returns"]
    risks = instance["risks"]
    budget = instance["budget"]
    n = instance["n_assets"]
    if abs(sum(x > 0.5) - budget) > 0.5:
        return float("inf")
    net = sum((returns[i] - risks[i]) for i in range(n) if x[i] > 0.5)
    return -float(net)  # negate: QUBO minimizes


# ---------------------------------------------------------------------------
# Maximum Clique
# ---------------------------------------------------------------------------

def max_clique_generator(seed: int, n_variables: int) -> dict:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 9))  # n in [5, 8]
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.6:
                edges.add((i, j))
    return {"n_nodes": n, "edges": list(edges)}


def max_clique_oracle(x: np.ndarray, instance: dict) -> float:
    """Returns -(clique size) if selected nodes form a clique, else +inf."""
    n = instance["n_nodes"]
    edge_set = {(i, j) for i, j in instance["edges"]}
    selected = [i for i in range(n) if x[i] > 0.5]
    for idx, i in enumerate(selected):
        for j in selected[idx + 1:]:
            pair = (min(i, j), max(i, j))
            if pair not in edge_set:
                return float("inf")
    return -float(len(selected))


# ---------------------------------------------------------------------------
# Minimum Dominating Set
# ---------------------------------------------------------------------------

def dominating_set_generator(seed: int, n_variables: int) -> dict:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 9))  # n in [5, 8]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                edges.append((i, j))
    # Build adjacency list
    neighbors = {i: [] for i in range(n)}
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    return {"n_nodes": n, "edges": edges, "neighbors": neighbors}


def dominating_set_oracle(x: np.ndarray, instance: dict) -> float:
    """Returns dominating set size if every node is dominated, else +inf."""
    n = instance["n_nodes"]
    neighbors = instance["neighbors"]
    for i in range(n):
        dominated = x[i] > 0.5 or any(x[nb] > 0.5 for nb in neighbors[i])
        if not dominated:
            return float("inf")
    return float(sum(x > 0.5))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "max_cut": {
        "file": "benchmarks/max_cut.txt",
        "generator": max_cut_generator,
        "oracle": max_cut_oracle,
    },
    "number_partition": {
        "file": "benchmarks/number_partition.txt",
        "generator": number_partition_generator,
        "oracle": number_partition_oracle,
    },
    "knapsack": {
        "file": "benchmarks/knapsack.txt",
        "generator": knapsack_generator,
        "oracle": knapsack_oracle,
    },
    "set_cover": {
        "file": "benchmarks/set_cover.txt",
        "generator": set_cover_generator,
        "oracle": set_cover_oracle,
    },
    "tsp_small": {
        "file": "benchmarks/tsp_small.txt",
        "generator": tsp_generator,
        "oracle": tsp_oracle,
    },
    "graph_coloring": {
        "file": "benchmarks/graph_coloring.txt",
        "generator": graph_coloring_generator,
        "oracle": graph_coloring_oracle,
    },
    "maximum_independent_set": {
        "file": "benchmarks/maximum_independent_set.txt",
        "generator": mis_generator,
        "oracle": mis_oracle,
    },
    "weighted_vertex_cover": {
        "file": "benchmarks/weighted_vertex_cover.txt",
        "generator": vertex_cover_generator,
        "oracle": vertex_cover_oracle,
    },
    "portfolio_optimization": {
        "file": "benchmarks/portfolio_optimization.txt",
        "generator": portfolio_generator,
        "oracle": portfolio_oracle,
    },
    "max_clique": {
        "file": "benchmarks/max_clique.txt",
        "generator": max_clique_generator,
        "oracle": max_clique_oracle,
    },
    "minimum_dominating_set": {
        "file": "benchmarks/minimum_dominating_set.txt",
        "generator": dominating_set_generator,
        "oracle": dominating_set_oracle,
    },
}
