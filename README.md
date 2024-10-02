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
Download pre-computed counts from https://drive.google.com/file/d/1At9V8Bw0xOOGoo7XFtBeIL60L5OJwk81/view?usp=drive_link
unzip and place them under `rooted_hom_count/tmp`

### Run node and graph classification experiments
* The scripts are in the `run` folder
* Run a single experiement
  * Graph classification
    `PYTHONPATH=../ python main.py --cfg configs/graph.yaml --repeat 1 # node classification`
  * Optinoall change `configs/graph.yaml` to run on other datasets and different patterns
* Run batch experiement
  * Graph classification
    `sh run_batch_graph.sh`
  * Optionally we can pass a different grid file by change the code in `run_batch_*.sh`
* The best performing model will be saved and used in the following steps
* We use graphgym to manage batch experiments, for advanced uses please refer to https://github.com/snap-stanford/GraphGym/tree/master
 
### Compute 1-WL/F-WL graph embeddings
* Uncomment line 601 in `PYTHONPATH=../ python compute_bound.py` in the `wl` folder
* Change the dataset and patterns in `save_lambda_features` accordingly
* Run `PYTHONPATH=../ python compute_bound.py`

### Compute generalisation bounds
* Uncomment line 602 in `PYTHONPATH=../ python compute_bound.py` in the `wl` folder
* Change the dataset and patterns in `print_bound` accordingly
* Make sure the corresponding model is saved in `models` folder and 1-WL/F-WL embeddings are computed
* Run `PYTHONPATH=../ python compute_bound.py`

### Compute homomorphism counts
Homomorphism and subgraph counts are pre-computed and saved under `rooted_hom_count/tmp`.
To re-compute them, following `README.md` in `rooted_hom_count` and run `compute_hom.py` or `compute_subgraph.py` in `rooted_hom_count` folder.