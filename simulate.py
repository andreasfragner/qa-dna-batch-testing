#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __LICENSE__
"""
Summary
-------
Simulate

Usage
-----
..code::

    python simulate.py --num-microplates 10000   \
                       --shape '(8, 12)'         \
                       --prevalence 0.1          \
                       --num-controls 6          \
                       --pos-controls 'top-left' \

    python simulate.py --help


References
----------
Beylerian et al, _Statistical Modeling for Quality Assurance of Human Papillomavirus
DNA Batch Testing_ (2018) (https://pubmed.ncbi.nlm.nih.gov/29570137/)
"""

__author__ = "andreasfragner"
__contributors__ = [__author__]

import argparse
import logging
from ast import literal_eval
from multiprocessing import cpu_count
from typing import Optional, Tuple

import numpy as np
import yaml
from scipy import ndimage


def analyze_clusters(microplate: np.ndarray, min_size: int = 2) -> Tuple:
    """
    Analyze cell clusters on a microplate

    Clusters are defined as groups of `min_size` or more horizontally or vertically
    adjacent positive cells

    Parameters
    ----------
    microplate : np.ndarray
        a 2D binary matrix representing a microplate assay

    min_size : int
        minimum number of horizontally or vertically adjacent positive cells to count
        as a cluster

    Returns
    -------
    tuple
        (total number of positive cells, number of clusters, size of largest cluster)
    """
    matrix = np.nan_to_num(microplate, 0)
    structure = ndimage.generate_binary_structure(2, 1)

    clusters, num_clusters = ndimage.label(matrix, structure)
    cluster_sizes = ndimage.sum(matrix, clusters, range(0, num_clusters + 1))

    mask = (cluster_sizes >= min_size)[clusters]
    clusters = clusters[mask]

    unique_labels, label_counts = np.unique(clusters, return_counts=True)

    return int(np.sum(matrix)), len(unique_labels), label_counts.max()


class Simulator:
    """
    Simulate microplate assay trials
    """

    CONTROL_POSITONS = ["top-left", "bottom-right"]

    def __init__(
        self,
        num_microplates: int,
        shape: tuple,
        prevalence: float,
        num_controls: int,
        pos_controls: str,
        num_processes: Optional[int] = None,
    ):

        self.logger = logging.getLogger(self.__class__.__name__)

        self.num_microplates = num_microplates
        self.num_controls = num_controls

        if not (prevalence > 0 and prevalence <= 1):
            raise ValueError(f"Expected a probability in [0, 1], got {prevalence}.")
        self.prevalence = prevalence

        if len(shape) != 2:
            raise ValueError(f"Expected tuple of length 2, got '{shape}'.")
        self.shape = shape

        if pos_controls not in self.CONTROL_POSITONS:
            raise ValueError(
                f"Expected one of {self.CONTROL_POSITONS}, " f"got {pos_controls}."
            )
        self.pos_controls = pos_controls

    def simulate(seed: Optional[int] = None):
        pass


class TupleParser(argparse.Action):
    """
    Argparse handler for parsing a str to a Python 2-tuple
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str,
        option_string=None,
    ):

        try:
            result = tuple(literal_eval(values))
        except ValueError:
            raise ValueError(f"Failed to parse {values} to 2-tuple")

        setattr(namespace, self.dest, result)


class Interface:
    """
    Command line interface for simulator
    """

    def __init__(self):
        self.name = self.__class__.__name__
        self.parser = argparse.ArgumentParser(
            prog=self.name,
            add_help=True,
            description=self.__doc__ + "\n" + __doc__,
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self.parser.add_argument(
            "--num-microplates",
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
            "--num-controls",
            type=int,
            required=True,
            help="the number of control wells on each microplate",
        )
        self.parser.add_argument(
            "--pos-controls",
            choice=["top-left", "bottom-right"],
            default="top-left",
            required=False,
            help="the position of the control wells on each microplate "
            "(default: %(default)s",
        )
        self.parser.add_argument(
            "--num-processes",
            type=int,
            default=cpu_count(),
            required=False,
            help="Number of processes for parallel simulation (default: %(default)s)",
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            default=None,
            required=False,
            help="passed to np.random.seed (default: %(default)s)",
        )

        self.parser.add_argument(
            "--logging-config",
            default="logging.yaml",
            type=str,
            required=False,
            help="Path to yaml logging config (default: %(default)s) ",
        )
        self.logger = logging.getLogger(self.name)

    def run(self):
        args = self.parser.parse_args()
        self.configure_logging(args)

        params = dict(args)
        seed = params.pop("seed")

        simulator = Simulator(**params)
        simulator.simulate(seed)

    @staticmethod
    def configure_logging(args: argparse.Namespace):
        with open(args.logging_config, "r") as fobj:
            cfg = yaml.load(fobj, Loader=yaml.SafeLoader)
        logging.config.dictConfig(cfg)


if __name__ == "__main__":
    app = Interface()
    app.run()
