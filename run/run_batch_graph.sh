#!/usr/bin/env bash

CONFIG=${CONFIG:-graph}
GRID=${GRID:-hom}
REPEAT=${REPEAT:-5}
MAX_JOBS=${MAX_JOBS:-5}
SLEEP=${SLEEP:-1}
MAIN=${MAIN:-main}

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python3 configs_gen.py --config configs/${CONFIG}.yaml \
  --grid grids/${GRID}.txt \
  --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running, sleep time
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN

# aggregate results for the batch
python3 agg_batch.py --dir results/${CONFIG}_grid_${GRID}
