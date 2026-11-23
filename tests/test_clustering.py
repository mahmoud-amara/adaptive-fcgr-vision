"""Unit and invariant tests for MinHash and Mash ANI clustering."""

import numpy as np
import pytest

from adaptive_cgr_vision.data.clustering import (
    compute_minhash_sketch,
    estimate_mash_ani,
    cluster_by_ani,
)


def test_minhash_identical_sequences():
    seq = "ACGTACGTACGTACGTACGT" * 100
    sk1 = compute_minhash_sketch(seq, k=11, sketch_size=50)
    sk2 = compute_minhash_sketch(seq, k=11, sketch_size=50)
    assert np.array_equal(sk1, sk2)
    ani = estimate_mash_ani(sk1, sk2, k=11)
    assert np.isclose(ani, 1.0)


def test_minhash_divergent_sequences():
    # Sequence A: AT rich
    seq_a = "ATATATAT" * 150
    # Sequence B: GC rich
    seq_b = "GCGCGCGC" * 150
    sk_a = compute_minhash_sketch(seq_a, k=7, sketch_size=50)
    sk_b = compute_minhash_sketch(seq_b, k=7, sketch_size=50)
    ani = estimate_mash_ani(sk_a, sk_b, k=7)
    assert ani < 0.2


def test_cluster_by_ani_grouping():
    # 2 variants of Sequence 1 (near identical)
    seq1_a = "ACGTACGTACGT" * 100
    seq1_b = "ACGTACGTACGT" * 99 + "ACGTACGTACGA"  # 1 base mutation

    # 1 divergent sequence
    seq2 = "TTTTGGGGCCCC" * 100

    seqs = [seq1_a, seq1_b, seq2]
    accs = ["ACC_1A", "ACC_1B", "ACC_2"]

    clusters = cluster_by_ani(seqs, accs, ani_threshold=0.90, k=9, sketch_size=50)
    assert len(clusters) == 3
    # 1A and 1B should share the same cluster
    assert clusters[0] == clusters[1]
    # 2 should be in a separate cluster
    assert clusters[2] != clusters[0]
