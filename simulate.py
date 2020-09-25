#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __LICENSE__
"""
Simulate distribution of test results from a microplate assay, following Ref. [1]

Usage
-----

python simulate.py --output '/path/to/results.csv'
                   --microplates 10000
                   --shape '(8, 12)'
                   --prevalence 0.16
                   --controls 6
                   --controls-position 'top-left'

References
----------
[1] Beylerian et al, _Statistical Modeling for Quality Assurance of Human Papillomavirus
DNA Batch Testing_ (2018) (https://pubmed.ncbi.nlm.nih.gov/29570137/)
"""

__author__ = "andreasfragner"
__contributors__ = [__author__]

import argparse
import logging
import os
from ast import literal_eval
from multiprocessing import cpu_count
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage


def analyze_matrix(matrix: np.ndarray, min_size: int = 2) -> Tuple:
    """
    Analyze cell clusters on a microplate matrix

    Clusters are defined as groups of `min_size` or more horizontally or vertically
    adjacent positive cells

    Parameters
    ----------
    matrix : np.ndarray
        a 2D binary matrix representing a microplate assay

    min_size : int
        minimum number of horizontally or vertically adjacent positive cells to count
        as a cluster

    Returns
    -------
    tuple
        (total number of positive cells, number of clusters, size of largest cluster)
    """
    if matrix.size == 0:
        return (0, 0, 0)

    matrix = np.nan_to_num(matrix, 0)
    structure = ndimage.generate_binary_structure(2, 1)

    clusters, num_clusters = ndimage.label(matrix, structure)
    cluster_sizes = ndimage.sum(matrix, clusters, range(0, num_clusters + 1))

    mask = (cluster_sizes >= min_size)[clusters]
    clusters = clusters[mask]

    unique_labels, label_counts = np.unique(clusters, return_counts=True)
    max_cluster_size = label_counts.max() if label_counts.size > 0 else 0

    return int(np.sum(matrix)), len(unique_labels), max_cluster_size


def analyze_vector(vector, padding, shape):
    matrix = np.pad(vector, padding, constant_values=0)
    matrix = np.reshape(matrix, shape)
    return analyze_matrix(matrix)


def get_logger(level="INFO"):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class Simulator:
    """
    Simulate distribution of test results for binary microplate assays
    """

    CONTROL_POSITONS = ["top-left", "bottom-right"]
    OUTPUT_COLUMNS = ("num_positive_cells", "num_clusters", "max_cluster_size")

    def __init__(
        self,
        microplates: int,
        shape: tuple,
        prevalence: float,
        controls: int,
        controls_position: Optional[str] = "top-left",
        log_level: Optional[str] = "INFO",
    ):

        self.logger = get_logger(log_level)
        self.microplates = microplates
        self.controls = controls

        if not (prevalence > 0 and prevalence <= 1):
            raise ValueError(f"Expected prevalence in [0, 1], got {prevalence}.")
        self.prevalence = prevalence

        if len(shape) != 2:
            raise ValueError(f"Expected tuple of length 2, got '{shape}'.")
        self.shape = shape

        if controls_position == "top-left":
            self.padding = (controls, 0)
        elif controls_position == "bottom_right":
            self.padding = (0, controls)
        else:
            raise ValueError(
                f"Expected one of {self.CONTROL_POSITONS}, got {controls_position}."
            )
        self.controls_position = controls_position

    def simulate(
        self, seed: Optional[int] = None, processes: Optional[int] = 1
    ) -> pd.DataFrame:
        """
        Run simulation

        Returns
        -------
        pd.DataFrame
            with shape `(self.microplates, 3)`
        """

        if seed is not None:
            np.random.seed(seed)

        self.logger.info(
            "Simulating %i microplates with shape %s and %i control wells",
            self.microplates,
            self.shape,
            self.controls,
        )

        num_samples = self.microplates * (self.shape[0] * self.shape[1] - self.controls)
        self.logger.info(
            "Generating %i binomial samples with probability %.2f ...",
            num_samples,
            self.prevalence,
        )

        samples = np.random.binomial(1, self.prevalence, size=num_samples)
        vectors = np.split(samples, self.microplates)

        results = Parallel(n_jobs=processes, verbose=1)(
            delayed(analyze_vector)(vector, self.padding, self.shape)
            for vector in vectors
        )

        return pd.DataFrame(results, columns=self.OUTPUT_COLUMNS)


class TupleParser(argparse.Action):
    """
    Argparse handler for parsing a str to a Python 2-tuple
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        value: str,
        option_string=None,
    ):

        try:
            result = tuple(literal_eval(value))
        except ValueError:
            raise ValueError(f"Failed to parse {value} to 2-tuple")

        setattr(namespace, self.dest, result)


class NumProcessesParser(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        value: Union[str, int],
        option_string=None,
    ):
        if value == "auto":
            value = cpu_count()
        elif not isinstance(value, int):
            raise TypeError(
                f"Got invalid value for num_processes '{value}', must "
                "be integer or 'auto'."
            )
        setattr(namespace, self.dest, value)


class Interface:
    """
    Command line interface for simulator
    """

    def __init__(self):
        self.name = self.__class__.__name__
        self.parser = argparse.ArgumentParser(
            prog=self.name,
            add_help=True,
            description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self.parser.add_argument(
            "--microplates",
            type=int,
            required=True,
            help="the number of microplates to simulate",
        )
        self.parser.add_argument(
            "--shape",
            type=str,
            required=True,
            action=TupleParser,
            help="microplate shape, a string of the form `(nrows, ncols)`",
        )
        self.parser.add_argument(
            "--prevalence",
            type=float,
            required=True,
            help="population prevalence, binomial success probability",
        )
        self.parser.add_argument(
            "--controls",
            type=int,
            required=True,
            help="the number of control wells on each microplate",
        )
        self.parser.add_argument(
            "--controls-position",
            choices=["top-left", "bottom-right"],
            default="top-left",
            required=False,
            help="the position of the control wells on each microplate "
            "(default: %(default)s)",
        )
        self.parser.add_argument(
            "--output",
            type=os.path.abspath,
            required=True,
            help="Output filepath. Results are written to a csv file with columns "
            f"{Simulator.OUTPUT_COLUMNS} with one row per microplate.",
        )
        self.parser.add_argument(
            "--processes",
            action=NumProcessesParser,
            default=1,
            required=False,
            help="Number of processes for parallel simulation (default: %(default)s). "
            "Use 'auto' for number of CPUs.",
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            default=None,
            required=False,
            help="passed to np.random.seed (default: %(default)s)",
        )

        self.parser.add_argument(
            "--log-level",
            default="INFO",
            required=False,
            choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"],
            help="Log level (default %(default)s)",
        )
        self.logger = logging.getLogger(self.name)

    def run(self):
        args = self.parser.parse_args()

        params = vars(args)
        seed = params.pop("seed")
        processes = params.pop("processes")
        output = params.pop("output")

        simulator = Simulator(**params)
        results = simulator.simulate(seed=seed, processes=processes)

        simulator.logger.info("Writing results to file %s", output)
        results.to_csv(output, index_label='microplate_id')


if __name__ == "__main__":
    app = Interface()
    app.run()
