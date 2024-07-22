#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <queue>

#include <sycl/sycl.hpp>

constexpr int INF = std::numeric_limits<int>::max();
constexpr int V = 9;  // Number of vertices
constexpr int E = 14; // Number of edges
constexpr int DELTA = 4; // Bucket size

struct Edge {
    int src, dest, weight;
};

void deltaStepping(const std::vector<Edge>& edges, int src) {
    // Flattened graph representation -> cant use complex vector types like vector<vector<Edges>>
    std::vector<int> edge_src(E);
    std::vector<int> edge_dest(E);
    std::vector<int> edge_weight(E);
    for (int i = 0; i < E; i++) {
        edge_src[i] = edges[i].src;
        edge_dest[i] = edges[i].dest;
        edge_weight[i] = edges[i].weight;
    }

    std::vector<int> dist(V, INF);
    dist[src] = 0;

    sycl::queue queue;
    std::cout << "Running on "
              << queue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    std::vector<std::vector<int>> buckets((INF / DELTA) + 1);
    buckets[0].push_back(src);

    sycl::buffer<int, 1> edge_src_buf(edge_src.data(), edge_src.size());
    sycl::buffer<int, 1> edge_dest_buf(edge_dest.data(), edge_dest.size());
    sycl::buffer<int, 1> edge_weight_buf(edge_weight.data(), edge_weight.size());
    sycl::buffer<int, 1> dist_buf(dist.data(), dist.size());

    auto process_bucket = [&](std::vector<int>& bucket) {
        sycl::buffer<int, 1> bucket_buf(bucket.data(), bucket.size());

        queue.submit([&](sycl::handler& cgh) {
            auto edge_src_acc = edge_src_buf.get_access<sycl::access::mode::read>(cgh);
            auto edge_dest_acc = edge_dest_buf.get_access<sycl::access::mode::read>(cgh);
            auto edge_weight_acc = edge_weight_buf.get_access<sycl::access::mode::read>(cgh);
            auto dist_acc = dist_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto bucket_acc = bucket_buf.get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for<class relax_edges>(sycl::range<1>(bucket.size()), [=](sycl::id<1> idx) {
                int u = bucket_acc[idx];
                for (int i = 0; i < E; i++) {
                    if (edge_src_acc[i] == u) {
                        int v = edge_dest_acc[i];
                        int weight = edge_weight_acc[i];

                        if (dist_acc[u] != INF && dist_acc[u] + weight < dist_acc[v]) {
                            dist_acc[v] = dist_acc[u] + weight;
                            // Debug 
                            printf("Updating distance of vertex %d to %d\n", v, dist_acc[v]);
                        }
                    }
                }
            });
        }).wait();
    };

    for (int i = 0; i < buckets.size(); ++i) {
        while (!buckets[i].empty()) {
            std::vector<int> bucket = std::move(buckets[i]);
            process_bucket(bucket);

            for (int u : bucket) {
                for (int j = 0; j < E; j++) {
                    if (edge_src[j] == u) {
                        int v = edge_dest[j];
                        int weight = edge_weight[j];

                        if (dist[v] < INF) {
                            int new_bucket_idx = dist[v] / DELTA;
                            if (new_bucket_idx >= buckets.size()) {
                                buckets.resize(new_bucket_idx + 1);
                            }
                            buckets[new_bucket_idx].push_back(v);
                            // Debug 
                            printf("Adding vertex %d to bucket %d\n", v, new_bucket_idx);
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < V; ++i) {
        if (dist[i] == INF)
            std::cout << "Vertex " << i << " is unreachable from source\n";
        else
            std::cout << "Distance from source to vertex " << i << " is " << dist[i] << "\n";
    }
}

int main() {
    std::vector<Edge> edges = {
        {0, 1, 4}, {0, 7, 8}, {1, 2, 8}, {1, 7, 11},
        {2, 3, 7}, {2, 8, 2}, {2, 5, 4}, {3, 4, 9},
        {3, 5, 14}, {4, 5, 10}, {5, 6, 2}, {6, 7, 1},
        {6, 8, 6}, {7, 8, 7}
    };

    deltaStepping(edges, 0);

    return 0;
}