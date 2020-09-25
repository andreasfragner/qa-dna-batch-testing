# -*- coding: utf-8 -*-
# __LICENSE__
"""
Unittests
"""

__author__ = "andreasfragner"
__contributors__ = [__author__]


import numpy as np
import pytest
from simulate import Simulator, analyze_matrix


@pytest.mark.parametrize(
    "matrix, expected",
    [
        (np.array([[]]), (0, 0, 0)),
        (np.array([[0]]), (0, 0, 0)),
        (np.array([[1]]), (1, 0, 0)),
        (np.array([[0, 0]]), (0, 0, 0)),
        (np.array([[1, 0]]), (1, 0, 0)),
        (np.array([[1, 1]]), (2, 1, 2)),
        (np.array([[0, 0], [0, 0]]), (0, 0, 0)),
        (np.array([[1, 0], [0, 0]]), (1, 0, 0)),
        (np.array([[1, 0], [0, 1]]), (2, 0, 0)),
        (np.array([[1, 1], [0, 0]]), (2, 1, 2)),
        (np.array([[1, 1], [1, 0]]), (3, 1, 3)),
        (np.array([[1, 1], [1, 1]]), (4, 1, 4)),
        (np.array([[1, 0, 1], [1, 0, 1]]), (4, 2, 2)),
        (np.array([[1, 1, 1], [1, 0, 1]]), (5, 1, 5)),
        (np.array([[0, 1, 0], [0, 1, 0]]), (2, 1, 2)),
        (np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]), (4, 0, 0)),
        (np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]), (5, 0, 0)),
        (np.array([[1, 0, 1], [1, 0, 0], [1, 0, 1]]), (5, 1, 3)),
        (np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]]), (6, 2, 3)),
        (np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), (8, 1, 8)),
        (np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), (5, 1, 5)),
    ],
)
def test_analyze_matrix(matrix, expected):
    results = analyze_matrix(matrix)
    assert results == expected


@pytest.mark.parametrize(
    "microplates, shape, controls, controls_position, seed",
    [(1000, (10, 10), 6, "top-left", 123)],
)
@pytest.mark.parametrize("prevalence", list(np.linspace(0.1, 1.0, 10)))
def test_simulator(microplates, shape, prevalence, controls, controls_position, seed):
    # summary statistics for num postive cells and max_cluster_size should
    # monotonically increase with prevalence
    cols = ["num_positive_cells", "max_cluster_size"]

    sim1 = Simulator(microplates, shape, prevalence, controls, controls_position)
    results1 = sim1.simulate(seed=seed)

    sim2 = Simulator(microplates, shape, prevalence - 0.05, controls, controls_position)
    results2 = sim2.simulate(seed=seed)

    stats1 = results1.describe().T
    stats2 = results2.describe().T

    assert (stats1["mean"][cols] >= stats2["mean"][cols]).all()
    assert (stats1["25%"][cols] >= stats2["25%"][cols]).all()
    assert (stats1["75%"][cols] >= stats2["75%"][cols]).all()
