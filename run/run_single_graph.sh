#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
python main.py --cfg configs/graph.yaml --repeat 1 # node classification
#python compute_bound.py --cfg configs/pyg/example_link.yaml --repeat 3 # link prediction
#python compute_bound.py --cfg configs/pyg/graph.yaml --repeat 3 # graph classification
