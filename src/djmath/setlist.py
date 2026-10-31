"""Ordering tracks into a DJ set.

Planned contents
----------------
- track graph with edge cost = tempo gap + harmonic distance
- greedy nearest-neighbour ordering
- exact Held-Karp dynamic programming, O(n^2 2^n)
- heuristic improvement (e.g. 2-opt) and runtime benchmarks

Implemented in a later commit.
"""
