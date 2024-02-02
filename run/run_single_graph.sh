#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
PYTHONPATH=../hom_gen python main.py --cfg configs/graph.yaml --repeat 1 # node classification
