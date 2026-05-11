#include "ahc_pyannote.h"

#include <gtest/gtest.h>

#include <set>
#include <vector>

TEST(AhcPyannoteTest, PdistEuclideanUsesCondensedPairOrder) {
    const std::vector<std::vector<float>> embeddings = {
        {0.0f, 0.0f},
        {3.0f, 4.0f},
        {6.0f, 8.0f},
    };

    const auto distances = ahc_pyannote::pdist_euclidean(embeddings);

    ASSERT_EQ(distances.size(), 3u);
    EXPECT_FLOAT_EQ(distances[0], 5.0f);
    EXPECT_FLOAT_EQ(distances[1], 10.0f);
    EXPECT_FLOAT_EQ(distances[2], 5.0f);
}

TEST(AhcPyannoteTest, LinkageCentroidBuildsExpectedSimpleDendrogram) {
    const auto dendrogram = ahc_pyannote::linkage_centroid({4.0f, 25.0f, 9.0f}, 3);

    ASSERT_EQ(dendrogram.size(), 2u);
    EXPECT_EQ(dendrogram[0].cluster_a, 0);
    EXPECT_EQ(dendrogram[0].cluster_b, 1);
    EXPECT_FLOAT_EQ(dendrogram[0].distance, 2.0f);
    EXPECT_EQ(dendrogram[0].size, 2);

    EXPECT_EQ(dendrogram[1].cluster_a, 2);
    EXPECT_EQ(dendrogram[1].cluster_b, 3);
    EXPECT_FLOAT_EQ(dendrogram[1].distance, 4.0f);
    EXPECT_EQ(dendrogram[1].size, 3);
}

TEST(AhcPyannoteTest, FclusterDistanceCutsTreeByStrictThreshold) {
    const auto dendrogram = ahc_pyannote::linkage_centroid({4.0f, 25.0f, 9.0f}, 3);

    const auto partial = ahc_pyannote::fcluster_distance(dendrogram, 3, 2.1f);
    ASSERT_EQ(partial.size(), 3u);
    EXPECT_EQ(partial[0], partial[1]);
    EXPECT_NE(partial[0], partial[2]);

    const auto merged = ahc_pyannote::fcluster_distance(dendrogram, 3, 4.1f);
    ASSERT_EQ(merged.size(), 3u);
    EXPECT_EQ(merged[0], merged[1]);
    EXPECT_EQ(merged[1], merged[2]);
}

TEST(AhcPyannoteTest, AgglomerativeClusterPyannoteHandlesEmptyAndSingletonInputs) {
    EXPECT_TRUE(ahc_pyannote::agglomerative_cluster_pyannote({}, 0.7045f, 12, 1, 1, -1).empty());

    const auto single = ahc_pyannote::agglomerative_cluster_pyannote({{1.0f, 0.0f}},
                                                                      0.7045f,
                                                                      12,
                                                                      1,
                                                                      1,
                                                                      -1);
    EXPECT_EQ(single, (std::vector<int>{0}));
}

TEST(AhcPyannoteTest, AgglomerativeClusterPyannoteHonorsRequestedSeparatedClusterCount) {
    const std::vector<std::vector<float>> embeddings = {
        {0.00f, 0.00f},
        {0.05f, 0.00f},
        {0.00f, 0.05f},
        {10.00f, 10.00f},
        {10.05f, 10.00f},
        {10.00f, 10.05f},
    };

    const auto labels = ahc_pyannote::agglomerative_cluster_pyannote(embeddings,
                                                                     0.7045f,
                                                                     12,
                                                                     2,
                                                                     2,
                                                                     2);

    ASSERT_EQ(labels.size(), embeddings.size());
    EXPECT_EQ(labels[0], labels[1]);
    EXPECT_EQ(labels[1], labels[2]);
    EXPECT_EQ(labels[3], labels[4]);
    EXPECT_EQ(labels[4], labels[5]);
    EXPECT_NE(labels[0], labels[3]);

    const std::set<int> unique_labels(labels.begin(), labels.end());
    EXPECT_EQ(unique_labels.size(), 2u);
}

TEST(AhcPyannoteTest, AgglomerativeClusterPyannoteReassignsSingletonToNearestLargeCluster) {
    const std::vector<std::vector<float>> embeddings = {
        {1.00f, 0.00f},
        {1.02f, 0.01f},
        {0.99f, -0.02f},
        {1.01f, 0.03f},
        {0.98f, 0.02f},
        {1.03f, -0.01f},
        {0.97f, 0.00f},
        {1.04f, 0.02f},
        {0.96f, -0.01f},
        {1.01f, -0.03f},
        {0.00f, 1.00f},
        {0.02f, 1.01f},
        {-0.01f, 0.98f},
        {0.03f, 1.02f},
        {0.01f, 0.97f},
        {-0.02f, 1.03f},
        {0.00f, 0.96f},
        {0.02f, 1.04f},
        {-0.03f, 0.99f},
        {2.00f, 0.30f},
    };

    const auto labels = ahc_pyannote::agglomerative_cluster_pyannote(embeddings,
                                                                     0.7045f,
                                                                     12,
                                                                     1,
                                                                     static_cast<int>(embeddings.size()),
                                                                     -1);

    ASSERT_EQ(labels.size(), embeddings.size());
    EXPECT_EQ(labels[19], labels[0]);
    EXPECT_NE(labels[19], labels[10]);

    const std::set<int> unique_labels(labels.begin(), labels.end());
    EXPECT_EQ(unique_labels.size(), 2u);
}
