// ahc_pyannote.cpp
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

#include "ahc_pyannote.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace ahc_pyannote {

namespace {

// Index into a condensed distance matrix for pair (i, j) with i != j.
// Symmetric in (i, j); i < j is enforced internally.
inline size_t condensed_index(int i, int j, int n) {
    if (i > j) std::swap(i, j);
    // i < j is now guaranteed
    return static_cast<size_t>(i) * static_cast<size_t>(n) -
           static_cast<size_t>(i) * static_cast<size_t>(i + 1) / 2 +
           static_cast<size_t>(j - i - 1);
}

}  // namespace


// =====================================================================
// pdist_euclidean
// =====================================================================

std::vector<float> pdist_euclidean(
    const std::vector<std::vector<float>>& embeddings) {
    const int n = static_cast<int>(embeddings.size());
    if (n < 2) return {};
    const size_t dim = embeddings.front().size();

    const size_t num_pairs = static_cast<size_t>(n) *
                             static_cast<size_t>(n - 1) / 2;
    std::vector<float> result(num_pairs);

    size_t idx = 0;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float sum_sq = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                const float diff = embeddings[i][d] - embeddings[j][d];
                sum_sq += diff * diff;
            }
            result[idx++] = std::sqrt(sum_sq);
        }
    }
    return result;
}


// =====================================================================
// linkage_centroid
// =====================================================================

// Naive O(N^3) Lance-Williams centroid linkage.
//
// Works on SQUARED distances. Lance-Williams centroid update rule:
//   d^2(uv, w) = (|u|/(|u|+|v|)) * d^2(u,w)
//              + (|v|/(|u|+|v|)) * d^2(v,w)
//              - (|u|*|v|/(|u|+|v|)^2) * d^2(u,v)
//
// On each merge: find the closest active pair, merge them, update
// distances from the new cluster to all remaining clusters.
//
// Cluster IDs follow the scipy convention:
//   - 0..n-1: original observations
//   - n..2n-2: merged clusters (in order of creation)
std::vector<DendrogramRow> linkage_centroid(
    std::vector<float> condensed_squared,
    int num_observations) {
    const int n = num_observations;
    if (n < 2) return {};

    std::vector<DendrogramRow> dendrogram;
    dendrogram.reserve(n - 1);

    // Full NxN squared-distance matrix kept during merging; inactive
    // clusters are marked in `active`. O(N^2) memory: for N~1000 this
    // is ~4MB, fine.
    const int max_clusters = 2 * n - 1;
    std::vector<float> dist_sq(static_cast<size_t>(max_clusters) *
                               static_cast<size_t>(max_clusters),
                               std::numeric_limits<float>::infinity());

    // Initial distances from condensed form
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const float d = condensed_squared[condensed_index(i, j, n)];
            dist_sq[static_cast<size_t>(i) * max_clusters + j] = d;
            dist_sq[static_cast<size_t>(j) * max_clusters + i] = d;
        }
    }

    std::vector<bool> active(max_clusters, false);
    std::vector<int> sizes(max_clusters, 1);
    for (int i = 0; i < n; ++i) {
        active[i] = true;
    }

    for (int merge_step = 0; merge_step < n - 1; ++merge_step) {
        // Find the closest active pair
        float min_dist_sq = std::numeric_limits<float>::infinity();
        int best_i = -1;
        int best_j = -1;
        const int total_clusters = n + merge_step;
        for (int i = 0; i < total_clusters; ++i) {
            if (!active[i]) continue;
            for (int j = i + 1; j < total_clusters; ++j) {
                if (!active[j]) continue;
                const float d = dist_sq[static_cast<size_t>(i) * max_clusters + j];
                if (d < min_dist_sq) {
                    min_dist_sq = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i < 0 || best_j < 0) {
            throw std::runtime_error("linkage_centroid: no active pairs found");
        }

        // New cluster gets ID = n + merge_step
        const int new_id = n + merge_step;
        const int size_i = sizes[best_i];
        const int size_j = sizes[best_j];
        const int new_size = size_i + size_j;
        sizes[new_id] = new_size;

        // Lance-Williams centroid update against all remaining active clusters
        const float total_inv = 1.0f / static_cast<float>(new_size);
        const float w_i = static_cast<float>(size_i) * total_inv;
        const float w_j = static_cast<float>(size_j) * total_inv;
        const float w_ij = w_i * w_j;  // = |i|*|j|/(|i|+|j|)^2
        const float d_ij_sq = min_dist_sq;

        for (int k = 0; k < total_clusters; ++k) {
            if (k == best_i || k == best_j || !active[k]) continue;
            const float d_ik_sq = dist_sq[static_cast<size_t>(best_i) * max_clusters + k];
            const float d_jk_sq = dist_sq[static_cast<size_t>(best_j) * max_clusters + k];
            const float new_d_sq = w_i * d_ik_sq + w_j * d_jk_sq - w_ij * d_ij_sq;
            // numerical floor against negative values from rounding
            const float clamped = std::max(0.0f, new_d_sq);
            dist_sq[static_cast<size_t>(new_id) * max_clusters + k] = clamped;
            dist_sq[static_cast<size_t>(k) * max_clusters + new_id] = clamped;
        }

        active[best_i] = false;
        active[best_j] = false;
        active[new_id] = true;

        // scipy convention: write the smaller cluster ID first
        dendrogram.push_back({
            std::min(best_i, best_j),
            std::max(best_i, best_j),
            std::sqrt(std::max(0.0f, min_dist_sq)),
            new_size,
        });
    }

    return dendrogram;
}


// =====================================================================
// fcluster_distance
// =====================================================================

// Cut the dendrogram at height `threshold`. Any subtree whose root merge
// distance is below `threshold` becomes one cluster; observations not
// covered by such a subtree become singleton clusters.
std::vector<int> fcluster_distance(
    const std::vector<DendrogramRow>& dendrogram,
    int num_observations,
    float threshold) {
    const int n = num_observations;
    if (n == 0) return {};
    if (n == 1) return {0};

    std::vector<int> labels(n, -1);

    // Iterative DFS from the root (last merge).
    struct StackEntry { int node; int cluster_label; };
    std::vector<StackEntry> stack;
    stack.reserve(2 * n);

    int next_label = 0;
    const int root = n + static_cast<int>(dendrogram.size()) - 1;
    stack.push_back({root, -1});  // -1 = no cluster assigned yet

    while (!stack.empty()) {
        const auto entry = stack.back();
        stack.pop_back();
        const int node = entry.node;
        int label = entry.cluster_label;

        if (node < n) {
            // Leaf node
            if (label < 0) {
                label = next_label++;
            }
            labels[node] = label;
            continue;
        }

        // Inner node
        const auto& row = dendrogram[node - n];
        if (label < 0 && row.distance < threshold) {
            // This subtree becomes one cluster
            label = next_label++;
        }

        stack.push_back({row.cluster_a, label});
        stack.push_back({row.cluster_b, label});
    }

    return labels;
}


// =====================================================================
// Internal helpers for agglomerative_cluster_pyannote
// =====================================================================

namespace {

// Compute the centroid (mean) of all embeddings with the given label.
std::vector<float> compute_centroid(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<int>& labels,
    int target_label) {
    if (embeddings.empty()) return {};
    const size_t dim = embeddings.front().size();
    std::vector<float> centroid(dim, 0.0f);
    int count = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == target_label) {
            for (size_t d = 0; d < dim; ++d) {
                centroid[d] += embeddings[i][d];
            }
            ++count;
        }
    }
    if (count > 0) {
        const float inv = 1.0f / static_cast<float>(count);
        for (float& v : centroid) v *= inv;
    }
    return centroid;
}


// Cosine distance between two vectors.
float cosine_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    const float denom = std::sqrt(na) * std::sqrt(nb) + 1e-10f;
    return 1.0f - dot / denom;
}


// Reapply fcluster at an arbitrary merge iteration. scipy does this by
// replacing Z[:,2] with arange(num_embeddings-1) and cutting at
// `iteration` as distance. Equivalent logic: a subtree is one cluster
// if its inner-node index < iteration.
std::vector<int> fcluster_at_iteration(
    const std::vector<DendrogramRow>& dendrogram,
    int num_observations,
    int iteration) {
    const int n = num_observations;
    if (n == 0) return {};
    if (n == 1) return {0};

    std::vector<int> labels(n, -1);
    struct StackEntry { int node; int cluster_label; };
    std::vector<StackEntry> stack;
    stack.reserve(2 * n);

    int next_label = 0;
    const int root = n + static_cast<int>(dendrogram.size()) - 1;
    stack.push_back({root, -1});

    while (!stack.empty()) {
        const auto entry = stack.back();
        stack.pop_back();
        const int node = entry.node;
        int label = entry.cluster_label;

        if (node < n) {
            if (label < 0) label = next_label++;
            labels[node] = label;
            continue;
        }

        const int inner_idx = node - n;  // merge iteration index
        const auto& row = dendrogram[inner_idx];
        // Cut rule: form a cluster when the merge happened before `iteration`
        if (label < 0 && inner_idx < iteration) {
            label = next_label++;
        }
        stack.push_back({row.cluster_a, label});
        stack.push_back({row.cluster_b, label});
    }

    return labels;
}


// Count occurrences of each cluster label.
std::unordered_map<int, int> cluster_sizes(const std::vector<int>& labels) {
    std::unordered_map<int, int> sizes;
    for (const int l : labels) sizes[l]++;
    return sizes;
}


// Count clusters whose size is >= min_cluster_size.
int count_large_clusters(
    const std::unordered_map<int, int>& sizes,
    int min_cluster_size) {
    int count = 0;
    for (const auto& [k, v] : sizes) {
        if (v >= min_cluster_size) ++count;
    }
    return count;
}

}  // namespace


// =====================================================================
// agglomerative_cluster_pyannote
// =====================================================================

std::vector<int> agglomerative_cluster_pyannote(
    const std::vector<std::vector<float>>& embeddings,
    float threshold,
    int min_cluster_size_param,
    int min_clusters,
    int max_clusters,
    int num_clusters) {
    const int num_embeddings = static_cast<int>(embeddings.size());
    if (num_embeddings == 0) return {};
    if (num_embeddings == 1) return {0};

    // Heuristic: relax min_cluster_size when there are few embeddings.
    // Mirrors pyannote's behavior.
    const int min_cluster_size = std::min(
        min_cluster_size_param,
        std::max(1, static_cast<int>(std::round(0.1f * num_embeddings))));

    // 1. Linkage
    auto condensed = pdist_euclidean(embeddings);
    // condensed holds euclidean distances; linkage_centroid expects squared
    for (float& d : condensed) d = d * d;
    auto dendrogram = linkage_centroid(std::move(condensed), num_embeddings);

    // 2. fcluster at the configured threshold
    auto clusters = fcluster_distance(dendrogram, num_embeddings, threshold);
    auto sizes = cluster_sizes(clusters);
    int num_large = count_large_clusters(sizes, min_cluster_size);

    // 3. Reconcile against min_clusters / max_clusters / num_clusters
    int target_clusters = num_clusters;
    if (num_large < min_clusters) {
        target_clusters = min_clusters;
    } else if (num_large > max_clusters) {
        target_clusters = max_clusters;
    }

    // 4. If a target cluster count is set but not reached at the
    //    threshold, walk the dendrogram looking for a cut that matches.
    if (target_clusters > 0 && num_large != target_clusters) {
        // Visit iterations in order of |distance - threshold|
        std::vector<int> iter_order(dendrogram.size());
        std::iota(iter_order.begin(), iter_order.end(), 0);
        std::sort(iter_order.begin(), iter_order.end(),
                  [&](int a, int b) {
                      return std::abs(dendrogram[a].distance - threshold) <
                             std::abs(dendrogram[b].distance - threshold);
                  });

        int best_iteration = num_embeddings - 1;
        int best_num_large = 1;
        std::vector<int> best_clusters = clusters;

        for (const int it : iter_order) {
            // Skip iterations whose resulting cluster is smaller than min_cluster_size
            if (dendrogram[it].size < min_cluster_size) continue;

            auto trial_clusters = fcluster_at_iteration(
                dendrogram, num_embeddings, it);
            auto trial_sizes = cluster_sizes(trial_clusters);
            int trial_num_large = count_large_clusters(trial_sizes, min_cluster_size);

            if (std::abs(trial_num_large - target_clusters) <
                std::abs(best_num_large - target_clusters)) {
                best_iteration = it;
                best_num_large = trial_num_large;
                best_clusters = std::move(trial_clusters);
            }

            if (trial_num_large == target_clusters) break;
        }

        clusters = std::move(best_clusters);
        sizes = cluster_sizes(clusters);
        num_large = best_num_large;
        (void)best_iteration;
    }

    if (num_large == 0) {
        // Collapse everything into a single cluster
        std::fill(clusters.begin(), clusters.end(), 0);
        return clusters;
    }

    // 5. Reassign each small cluster to its nearest large cluster by
    //    centroid cosine distance.
    std::vector<int> large_ids, small_ids;
    for (const auto& [k, v] : sizes) {
        (v >= min_cluster_size ? large_ids : small_ids).push_back(k);
    }
    std::sort(large_ids.begin(), large_ids.end());
    std::sort(small_ids.begin(), small_ids.end());

    if (!small_ids.empty()) {
        std::vector<std::vector<float>> large_centroids;
        large_centroids.reserve(large_ids.size());
        for (const int lid : large_ids) {
            large_centroids.push_back(compute_centroid(embeddings, clusters, lid));
        }

        for (const int sid : small_ids) {
            const auto small_centroid = compute_centroid(embeddings, clusters, sid);
            int best_large = large_ids[0];
            float best_dist = std::numeric_limits<float>::infinity();
            for (size_t li = 0; li < large_ids.size(); ++li) {
                const float d = cosine_distance(small_centroid, large_centroids[li]);
                if (d < best_dist) {
                    best_dist = d;
                    best_large = large_ids[li];
                }
            }
            for (int& lbl : clusters) {
                if (lbl == sid) lbl = best_large;
            }
        }
    }

    // 6. Renumber clusters to 0..K-1
    std::unordered_map<int, int> remap;
    int next_id = 0;
    for (int& lbl : clusters) {
        auto it = remap.find(lbl);
        if (it == remap.end()) {
            remap[lbl] = next_id++;
            lbl = next_id - 1;
        } else {
            lbl = it->second;
        }
    }

    return clusters;
}

}  // namespace ahc_pyannote