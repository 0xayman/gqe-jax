#!/bin/bash

echo "Starting job..."
python benchmark.py

echo "Job finished, shutting down..."
sudo shutdown -h now
