# Toy Example to test EMOGI
We provided a little toy example to test your local setup of EMOGI on a small-scale simulated dataset.
The network is a simulated random small-world graph with three cliques embedded. The features for the clique nodes are drawn from the known cancer genes while the non-clique node features are drawn from the non-cancer genes.
You may run the toy example simply using:
```
cd EMOGI
python train_EMOGI.py -e 2000 -s 1 -hd 100 -lm 30 -d ../example/toy_example.h5
```
These are hyper parameters that were not optimized using a grid search but are expected to work to some degree in practice.
The training is expected to take only some seconds or few minutes on a machine with 64 cores and no GPU.
