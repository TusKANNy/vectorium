#!/usr/bin/env bash
set -euo pipefail

ARGS_BASE="-i ../data/documents.bin -q ../data/queries.bin -k 10 \
           --warmup 5 --repeats 1 \
           --groundtruth-tsv ../data/groundtruth.tsv \
           --n-clusters 1024"

# clustered-centroid k-means, two points
perf stat -x , -d -d -d \
  -o stat_kmeans_clustered_n1.csv \
  -- ./target/release/bench_4bit_efficiency_kmeans_clustered $ARGS_BASE --n-queries 1

perf stat -x , -d -d -d \
  -o stat_kmeans_clustered_n100.csv \
  -- ./target/release/bench_4bit_efficiency_kmeans_clustered $ARGS_BASE --n-queries 100
