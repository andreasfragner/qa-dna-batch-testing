# -*- coding: utf-8 -*-
# __LICENSE__
"""
Unittests
"""

__author__ = 'andreasfragner'
__contributors__ = [__author__]


import pytest

from ..simulate import analyze_clusters


@pytest.mark.parametrize("microplate, expected", [])
def test_analyze_clusters(microplate, expected):
    results = analyze_clusters(microplate)
    assert results == expected
