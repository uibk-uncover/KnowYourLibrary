
import collections
import numpy as np
import sys

from . import mismatch
from .compression import CompressionTestResults
from .decompression import DecompressionTestResults


def print_intro(dataset: np.ndarray):
    print("Data:")
    print("| Data size:", dataset.shape[0])
    print("| Image size:", dataset.shape[1:3])
    print("| Channels:", dataset.shape[3], end="\n\n")
    sys.stdout.flush()


def print_clusters(clusters):
    if type(clusters) == CompressionTestResults:
        pass
    if type(clusters) == DecompressionTestResults:
        print('| spatial', clusters.spatial)
    sys.stdout.flush()


_joint = collections.OrderedDict()


def add_print_grouped_clusters(clusters, identifier):
    """Call me with clusters and identifier of a call. I will keep track of it and print it nicely for you."""
    global _joint
    # is empty
    was_empty = len(_joint) == 0
    # add to joint
    clusters = mismatch.order_cluster_group(clusters)
    if clusters not in _joint:
        _joint[clusters] = [identifier]
    else:
        _joint[clusters].append(identifier)
    # get the first
    k1 = next(iter(_joint))
    # print legend
    if was_empty:
        print("|", k1, ":", end="")
    # print until empty
    while _joint[k1]:
        print(" ", _joint[k1][0], sep="", end="")
        _joint[k1] = _joint[k1][1:]
    sys.stdout.flush()


def end_print_grouped_clusters():
    """Call me when done with add_print_grouped_clusters."""
    global _joint
    for k in _joint:
        # skip empty (first)
        if not _joint[k]:
            print()
            continue
        # print legend
        print("|", k, ":", end="")
        # print until empty
        while _joint[k]:
            print(" ", _joint[k][0], sep="", end="")
            _joint[k] = _joint[k][1:]
        print()
    _joint = collections.OrderedDict()
    sys.stdout.flush()
