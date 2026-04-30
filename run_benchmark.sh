#!/usr/bin/env bash
echo "Running benchmark.py -n 50 -q 4..."
python benchmark.py -n 50 -q 4

echo "Benchmark finished. Shutting down VM..."
sudo shutdown -h now
