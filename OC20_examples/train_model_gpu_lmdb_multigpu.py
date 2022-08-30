import numpy as np
import random
import torch
from ase import Atoms
from ase.calculators.emt import EMT
​
from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
import sys
import os
​
trail_num = sys.argv[1]
checkpoint_name = sys.argv[2]
​
nsigmas = int(sys.argv[3])
MCSHs_index = int(sys.argv[4])
NN_index = int(sys.argv[5])
num_data = int(sys.argv[6])
batch_size = int(sys.argv[7])
​
num_max_dataset = int(num_data / 50000)
lr = float(sys.argv[8])
epochs = int(sys.argv[9])
n_gpu = torch.cuda.device_count()
print("****\n Found {} GPUs \n****\n\n".format(n_gpu))
n_gpu = 0
#assert torch.cuda.device_count() >= 2
​
num_val = 5000
num_val_max_dataset = 1
#val_split = float(sys.argv[11])
val_split = num_val / (num_data + num_val)
​
​
print("total num data: {}".format(num_data))
print("batch size: {}".format(batch_size))
print("learning rate: {}".format(lr))
print("num epochs: {}".format(epochs))
print("val split: {}".format(val_split))
​
​
​
folder_name = "trial_{}".format(trail_num)
os.chdir(folder_name)
​
lmdb_paths = ["./lmdbs_sigma{}_MCSH{}_force_double/{}_{}.lmdb".format(nsigmas,MCSHs_index,i,j) for i in range(num_max_dataset) for j in range(100)]
random.Random(1).shuffle(lmdb_paths)
lmdb_paths += ["./lmdbs_sigma{}_MCSH{}_force_double/val_{}_{}.lmdb".format(nsigmas,MCSHs_index,i,j) for i in range(num_val_max_dataset) for j in range(10)]
​
NN_dict = {
1: [32,32,32],
2: [64,64,64],
3: [128,64,64],
4: [256,128,64],
5: [512,256,128,64],
6: [1024,512,128,64],
}
hidden_layers = NN_dict[NN_index]
​
config = {
    "model": {
        "name":"singlenn",
        "get_forces": True, 
        "hidden_layers": hidden_layers, 
        "activation":torch.nn.GELU,
        "batchnorm": True,
    },
    "optim": {
        "gpus":n_gpu,
        "force_coefficient": 3.0,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "loss": "mae",
        #"scheduler": {"policy": "StepLR", "params": {"step_size": 1000, "gamma": 0.5}}
    },
    "dataset": {
        "lmdb_path": lmdb_paths,
        "val_split": val_split,
        "val_split_mode": "inorder",
        "cache": "full"
    },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "ordernorm_gelu_sigma{}_MCSH{}_NN{}_lr{}_batch{}_ep{}_force".format(nsigmas, MCSHs_index,NN_index, lr, batch_size, epochs),
        "verbose": True,
        # Weights and Biases used for logging - an account(free) is required
        "logger": False,
        "dtype": torch.DoubleTensor
    },
}
​
trainer = AtomsTrainer(config)
if os.path.isdir(checkpoint_name):
    try:
        trainer.load_pretrained(checkpoint_name, gpu2cpu=True)
    except:
        trainer.load_pretrained(checkpoint_name)
else:
    print("**** WARNING: checkpoint not found: {} ****".format(checkpoint_name))
print("training")
trainer.train()
print("end training")