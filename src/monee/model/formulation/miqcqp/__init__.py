"""Mixed-integer QCQP formulations.

``convex``: relaxations a convex MIQCQP/MISOCP solver can certify
(branch-flow SOC for electricity, epigraph-relaxed Weymouth for gas).
``nonconvex``: the exact quadratic models (SOC/epigraph as equalities,
bilinear temperature transport) for global MIQCQP solvers like SCIP/Gurobi.
"""
