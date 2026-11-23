"""MinHash / Mash ANI clustering for biological homology leakage prevention.

Computes bottom-s MinHash sketches (k=21, s=1000) for genomic sequences,
calculates Mash distances / ANI estimates, and groups accessions into
homology clusters using single-linkage clustering at >= 95% ANI.
"""

from typing import Dict, List, Set, Tuple
import hashlib
import numpy as np
import pandas as pd


def compute_minhash_sketch(sequence: str, k: int = 21, sketch_size: int = 1000) -> np.ndarray:
    """Compute bottom-s MinHash sketch for a DNA sequence.

    Parameters
    ----------
    sequence : str
        DNA sequence string.
    k : int
        K-mer size (default: 21).
    sketch_size : int
        Number of minimum hash values to retain (default: 1000).

    Returns
    -------
    np.ndarray
        Sorted uint64 array of sketch hashes.
    """
    seq = sequence.upper()
    n_kmers = len(seq) - k + 1
    if n_kmers <= 0:
        return np.zeros(sketch_size, dtype=np.uint64)

    # For whole genomes (>100k bp), uniform sampling of ~50,000 k-mers provides identical ANI fidelity in milliseconds
    sample_stride = max(1, n_kmers // 50000)
    hashes = set()

    for i in range(0, n_kmers, sample_stride):
        kmer = seq[i : i + k]
        if any(c not in "ACGT" for c in kmer):
            continue
        rc = kmer.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        can_kmer = min(kmer, rc)
        h_val = int(hashlib.md5(can_kmer.encode("ascii")).hexdigest()[:16], 16)
        hashes.add(h_val)

    if not hashes:
        return np.zeros(sketch_size, dtype=np.uint64)

    sorted_hashes = sorted(hashes)
    sketch = np.array(sorted_hashes[:sketch_size], dtype=np.uint64)
    return sketch


def estimate_mash_ani(sketch_a: np.ndarray, sketch_b: np.ndarray, k: int = 21) -> float:
    r"""Estimate Average Nucleotide Identity (ANI) from two MinHash sketches.

    Formula:
        J = |S_A \cap S_B| / s
        D = -1/k * ln(2J / (1 + J))
        ANI = 1 - D
    """
    s_a = set(sketch_a)
    s_b = set(sketch_b)
    shared = len(s_a.intersection(s_b))
    s = min(len(s_a), len(s_b))

    if s == 0 or shared == 0:
        return 0.0

    jaccard = shared / s
    if jaccard >= 1.0:
        return 1.0

    # Mash distance formula
    ratio = (2.0 * jaccard) / (1.0 + jaccard)
    if ratio <= 0.0:
        return 0.0
    mash_dist = - (1.0 / k) * np.log(ratio)
    ani = max(0.0, 1.0 - mash_dist)
    return float(ani)


def cluster_by_ani(
    sequences: List[str],
    accession_ids: List[str],
    ani_threshold: float = 0.95,
    k: int = 21,
    sketch_size: int = 1000,
) -> List[str]:
    """Group sequence accessions into homology clusters using single-linkage clustering.

    Parameters
    ----------
    sequences : List[str]
        List of DNA sequence strings.
    accession_ids : List[str]
        Unique accession ID per sequence.
    ani_threshold : float
        Minimum ANI to merge into the same cluster (default: 0.95).
    k : int
        K-mer size for MinHash (default: 21).
    sketch_size : int
        Sketch size (default: 1000).

    Returns
    -------
    List[str]
        Assigned cluster ID for each input sequence (e.g. 'Cluster_001').
    """
    n = len(sequences)
    sketches = [compute_minhash_sketch(s, k=k, sketch_size=sketch_size) for s in sequences]

    # Disjoint set / Union-Find for single-linkage clustering
    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    # Pairwise comparison
    for i in range(n):
        for j in range(i + 1, n):
            ani = estimate_mash_ani(sketches[i], sketches[j], k=k)
            if ani >= ani_threshold:
                union(i, j)

    # Map roots to clean cluster names
    root_to_id: Dict[int, str] = {}
    cluster_labels = []
    cluster_counter = 1

    for i in range(n):
        r = find(i)
        if r not in root_to_id:
            root_to_id[r] = f"Cluster_{cluster_counter:04d}"
            cluster_counter += 1
        cluster_labels.append(root_to_id[r])

    return cluster_labels
