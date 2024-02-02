### Installation
* Update git submodules
`git submodule update --init --recursive`

Install the following packages
* PyTorch
* Pytorch-Geometric
* NetworkX

or simply run `pip install -r requirements.txt`
(you might want to remove cuda related packages if you don't have a GPU)

### Data preparation
Download pre-computed counts from https://github.com/snap-stanford/GraphGym/tree/master/run
unzip and place them under `rooted_hom_count/tmp`

### Compute data-dependent bounds
Run `compute_bounds.py` in the `wl` folder, results are saved in `fig/test.csv`
You can edit `compute_bounds.py` to compute bounds for other datasets. 

### Run node and graph classification experiments
* Run a single experiement
  * Graph classification
    `sh run_single_graph.sh`
  * Node classification
    `sh run_single_node.sh`
* Run batch experiement
  * Graph classification
    `sh run_batch_graph.sh`
  * Node classification
    `sh run_batch_node.sh`
  * Optionally we can pass a different grid file by change the code in `run_batch_*.sh`

We use graphgym to manage batch experiments, for advanced uses please refer to https://github.com/snap-stanford/GraphGym/tree/master


### Compute homomorphism counts
Homomorphism and subgraph counts are pre-computed and saved under `rooted_hom_count/tmp`.
To re-compute them, following `README,md` in `rooted_hom_count` and run `compute_hom.py` or `compute_subgraph.py` in `rooted_hom_count` folder.