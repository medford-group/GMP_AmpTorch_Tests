import numpy as np
import random
import torch
from ase import Atoms
from ase.calculators.emt import EMT

from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
import sys
import os

trail_num = sys.argv[1]
checkpoint_name = sys.argv[2]

sigmas_index = int(sys.argv[3])
MCSHs_index = int(sys.argv[4])
num_nodes = int(sys.argv[5])
num_layers = int(sys.argv[6])
batch_size = int(sys.argv[7])
num_data = int(sys.argv[8])
num_max_dataset = int(num_data / 5000)
lr = float(sys.argv[9])
epochs = int(sys.argv[10])

num_val = 50000
num_val_max_dataset = int(num_val / 5000)
#val_split = float(sys.argv[11])
val_split = num_val / (num_data + num_val)


print("total num data: {}".format(num_data))
print("batch size: {}".format(batch_size))
print("learning rate: {}".format(lr))
print("num epochs: {}".format(epochs))
print("val split: {}".format(val_split))



folder_name = "trial_{}".format(trail_num)
os.chdir(folder_name)

lmdb_paths = ["./lmdbs_sigma{}_MCSH{}/{}.lmdb".format(sigmas_index,MCSHs_index,i) for i in range(num_max_dataset)]
random.Random(1).shuffle(lmdb_paths)
lmdb_paths += ["./lmdbs_sigma{}_MCSH{}/val_{}.lmdb".format(sigmas_index,MCSHs_index,i) for i in range(num_val_max_dataset)]

config = {
    "model": {
        "name":"singlenn",
        "get_forces": False,
        "num_layers": num_layers, 
        "num_nodes": num_nodes,
        "batchnorm": True
    },
    "optim": {
        "gpus":1,
        #"force_coefficient": 0.04,
        "force_coefficient": 0.0,
        "lr": lr,
        "batch_size": batch_size,
        "epochs":epochs,
        "loss": "mae",
    },
    "dataset": {
        "lmdb_path": lmdb_paths,
        "val_split": val_split,
        "val_split_mode": "inorder",
        "cache": "partial"
    },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "test",
        "verbose": True,
        # Weights and Biases used for logging - an account(free) is required
        "logger": False,
    },
}

trainer = AtomsTrainer(config)
if os.path.isdir(checkpoint_name):
    trainer.load_pretrained(checkpoint_name)
else:
    print("**** WARNING: checkpoint not found: {} ****".format(checkpoint_name))
print("training")
trainer.train()
print("end training")
