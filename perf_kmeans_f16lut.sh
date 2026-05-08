#!/usr/bin/env bash
set -euo pipefail

ARGS_BASE="-i ../data/documents.bin -q ../data/queries.bin -k 10 \
           --warmup 5 --repeats 1 \
           --groundtruth-tsv ../data/groundtruth.tsv"

# f16-LUT k-means, two points
perf stat -x , -d -d -d \
  -o stat_kmeans_f16lut_n1.csv \
  -- ./target/release/bench_4bit_efficiency_kmeans_f16lut $ARGS_BASE --n-queries 1

perf stat -x , -d -d -d \
  -o stat_kmeans_f16lut_n100.csv \
  -- ./target/release/bench_4bit_efficiency_kmeans_f16lut $ARGS_BASE --n-queries 100
