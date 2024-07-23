# Parallel single-source shortest path algorithm 

<p align="center">
  <img src="https://imgs.xkcd.com/comics/space_typography.png" alt="Space Typography">
</p>

_Comic by Randall Munroe, [XKCD](https://xkcd.com/590/)_

In algorithmic graph theory, a fundamental challenge is the shortest path problem. A broader variant of this problem is the single-source shortest paths (SSSP) problem, which involves finding the shortest paths from a given source vertex $s$ to all other vertices in the graph. Traditionally, this problem is solved using classical sequential algorithms like Dijkstra's algorithm. However, I introduce a parallel algorithm designed to solve the SSSP problem know as Delta stepping algorithm.

## Delta stepping algorithm 

The delta stepping algorithm is designed to iteratively adjust vertex distances in a graph until they reach their final values. It does this by using a series of corrections throughout its steps. 

In this algorithm, vertices are placed into an array of buckets, with each bucket representing a specific range of distances determined by a parameter $\Delta$.  During each phase, the algorithm process vertices from the first non-empty bucket, relaxing their outgoing edges that have weights up to $\Delta$. Edges with weight greater than $\Delta$ are only relaxed later, after ensuring that the starting vertices have finished. The value $\Delta$, referred to as the step width or bucket width, is a positive number.  

To achieve parallelism, the algorithm removes all vertices from the first non-empty bucket simulatneously and relaxes their light edges. If a vertex $v$ is removed but does not yet have a final distance, it can be added back to the bucket, and its light edges will be relaxed again. Heavy edges are relaxed once the bucket is fully processed and becomes empty. The algorithm then proceeds to the next non-empty bucket and continues the process. 


The maximum shortest path weight for a source vertex $s$ is define as 

$$L(s) = \\max\\{dist(s, v) : dist(s, v) < \infty\\}$$

where, $dist(v)$ represents the distance between the source vertex $s$ and another vertex $v$, The size of a path if defined by the number of edges traversed in that path. Light edges are those with weights up to the value $\Delta$, while the heavy edges have weights greater than $\Delta$.


# Compile on Polaris 

If you have access to Polaris with the following directions you will be able to test the algorithm with a default graph that is indicated on the Parallel SSSP on [Wiki](https://en.wikipedia.org/wiki/Parallel_single-source_shortest_path_algorithm)

1. Start with loading modules that we need to correctly add all necessary depencedinces and oneapi.
   ```bash
    module use /soft/modulefiles
    module load oneapi/upstream
    module load nvhpc-mixed
    module load craype-accel-nvidia80
    module unload nvhpc-mixed
    module load spack-pe-base cmake
   ```
2. Add the following `FLAGS`
   ```bash
    EXTRA_FLAGS="-sycl-std=2020 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80"
    export CFLAGS="-ffp-model=precise"
    export CXXFLAGS="-ffp-model=precise -fsycl $EXTRA_FLAGS"
    export CC=clang
    export CXX=clang++
   ```
3. Clone the repo `git clone https://github.com/JorgeV92/SSSP.git`

Head into the directory `SSSP` and run the following commands 

- `cmake -DCMAKE_BUILD_TYPE=Debug -S . -B build` then `cmake --build build`
