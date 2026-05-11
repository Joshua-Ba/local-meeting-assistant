// ahc_pyannote.h
//
// C++ port of pyannote.audio's AgglomerativeClustering pipeline,
// using a centroid-linkage AHC algorithm modeled after
// scipy.cluster.hierarchy.linkage(method="centroid").
//
// References:
//   pyannote.audio (MIT License)
//     https://github.com/pyannote/pyannote-audio
//     pyannote/audio/pipelines/clustering.py - AgglomerativeClustering.cluster()
//
//   scipy (BSD-3-Clause License)
//     https://github.com/scipy/scipy
//     scipy/cluster/hierarchy.py - linkage(method="centroid")
//
// This implementation is independent C++ code; no Python source was
// directly copied. The algorithm itself follows established hierarchical
// clustering literature (Lance-Williams update for centroid linkage).

#pragma once

#include <vector>

namespace ahc_pyannote {

// One entry of the dendrogram: two cluster IDs are merged into a new
// cluster with the given distance and size.
// IDs < num_observations are original embeddings; IDs >= num_observations
// are clusters created by earlier merges (matches the scipy format).
struct DendrogramRow {
    int cluster_a;
    int cluster_b;
    float distance;
    int size;
};

// Compute pairwise euclidean distances.
// Returns a condensed distance matrix of length n*(n-1)/2.
// Index for pair (i, j) with i < j: i*n - i*(i+1)/2 + (j-i-1).
std::vector<float> pdist_euclidean(
    const std::vector<std::vector<float>>& embeddings);

// Lance-Williams centroid linkage.
// Input: condensed distance matrix of SQUARED distances (not euclidean!).
// Output: dendrogram of length n-1 with real euclidean distances.
//
// Working on squared form is critical: the Lance-Williams centroid update
// is defined on d^2; mixing in sqrt round-trips introduces precision loss.
std::vector<DendrogramRow> linkage_centroid(
    std::vector<float> condensed_squared,
    int num_observations);

// fcluster with criterion="distance": every subtree whose root merge
// distance is below `threshold` becomes one cluster.
// Output: cluster_labels[i] = 0-indexed cluster ID for observation i.
std::vector<int> fcluster_distance(
    const std::vector<DendrogramRow>& dendrogram,
    int num_observations,
    float threshold);

// Main entry point: pyannote-conformant AHC with all phases.
//
// Steps:
// 1. min_cluster_size heuristic (relaxed for few embeddings)
// 2. Compute linkage
// 3. fcluster at threshold
// 4. Classify clusters by size (large vs small)
// 5. If num_clusters is set and num_large != target:
//    - Walk dendrogram iterations ordered by |dist - threshold|
//    - Find an iteration where num_large == target
// 6. Reassign small clusters to nearest large cluster (centroid distance)
// 7. Renumber clusters to 0..K-1
//
// Returns: cluster_labels[i] in 0..K-1.
std::vector<int> agglomerative_cluster_pyannote(
    const std::vector<std::vector<float>>& embeddings,
    float threshold,
    int min_cluster_size,
    int min_clusters,
    int max_clusters,
    int num_clusters);  // -1 = not set

}  // namespace ahc_pyannote