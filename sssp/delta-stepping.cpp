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
    std::vector<std::vector<Edge>> adj(V);
    for (const auto& edge : edges) {
        adj[edge.src].push_back(edge);
    }

    std::vector<int> dist(V, INF);
    dist[src] = 0;

    cl::sycl::queue queue;
    std::cout << "Running on "
              << queue.get_device().get_info<cl::sycl::info::device::name>()
              << "\n";

    // Buckets
    std::vector<std::vector<int>> buckets((INF / DELTA) + 1);
    buckets[0].push_back(src);

    cl::sycl::buffer<int, 1> dist_buf(dist.data(), dist.size());

    auto process_bucket = [&](std::vector<int>& bucket) {
        cl::sycl::buffer<int, 1> bucket_buf(bucket.data(), bucket.size());

        queue.submit([&](cl::sycl::handler& cgh) {
            auto dist_acc = dist_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto bucket_acc = bucket_buf.get_access<cl::sycl::access::mode::read>(cgh);

            cgh.parallel_for<class relax_edges>(cl::sycl::range<1>(bucket.size()), [=](cl::sycl::id<1> idx) {
                int u = bucket_acc[idx];

                for (const auto& edge : adj[u]) {
                    int v = edge.dest;
                    int weight = edge.weight;

                    if (dist_acc[u] != INF && dist_acc[u] + weight < dist_acc[v]) {
                        int new_dist = dist_acc[u] + weight;
                        dist_acc[v] = new_dist;
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
                for (const auto& edge : adj[u]) {
                    int v = edge.dest;
                    int weight = edge.weight;

                    if (dist[v] < INF) {
                        int new_bucket_idx = dist[v] / DELTA;
                        if (new_bucket_idx >= buckets.size()) {
                            buckets.resize(new_bucket_idx + 1);
                        }
                        buckets[new_bucket_idx].push_back(v);
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
