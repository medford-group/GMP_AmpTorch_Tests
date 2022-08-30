# importing the module 
import json 
import ase
import numpy as np
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from amptorch.trainer import AtomsTrainer
import pickle
from ase.io.trajectory import Trajectory
from ase.io import read
import sys
import time
import torch
import os
import csv
from skorch.callbacks import LRScheduler
​
def log(log_filename, message):
    f = open(log_filename, "a")
    f.write(message)
    f.close()
    return
​
def construct_parameter_set(num_gaussian, max_MCSH_order, log_filename = "info.log"):
​
    sigmas = np.linspace(0.0,2.0,num_gaussian+1,endpoint=True)[1:].tolist()
​
​
    MCSHs_dict = {
        0: { "orders": [0], "sigmas": sigmas,},
        1: { "orders": [0,1], "sigmas": sigmas,},
        2: { "orders": [0,1,2], "sigmas": sigmas,},
        3: { "orders": [0,1,2,3], "sigmas": sigmas,},
        4: { "orders": [0,1,2,3,4], "sigmas": sigmas,},
        5: { "orders": [0,1,2,3,4,5], "sigmas": sigmas,},
        6: { "orders": [0,1,2,3,4,5,6], "sigmas": sigmas,},
        7: { "orders": [0,1,2,3,4,5,6,7], "sigmas": sigmas,},
        8: { "orders": [0,1,2,3,4,5,6,7,8], "sigmas": sigmas,},
        9: { "orders": [0,1,2,3,4,5,6,7,8,9], "sigmas": sigmas,},
    }
​
    MCSH_setup = MCSHs_dict[max_MCSH_order]
​
​
    MCSHs = {   "MCSHs": MCSH_setup,
            "atom_gaussians": {
                        "H": "../../psp_pseudo_v3/H_pseudodensity.g",
                        "C": "../../psp_pseudo_v3/C_pseudodensity.g",
                        "O": "../../psp_pseudo_v3/O_pseudodensity.g",
                  },
            "cutoff": 10.0, 
            "square":False,
            "solid_harmonics": True,
    }
​
    log(log_filename, json.dumps(MCSHs,indent = 4) + "\n")
    return MCSHs
​
def test_model(trainer, data_list, data_type = "test", log_filename = "info.log"):
    start = time.time()
    predictions = trainer.predict(data_list)
    time_took = time.time() - start
​
    true_energies = np.array([image.get_potential_energy() for image in data_list])
    pred_energies = np.array(predictions["energy"])
    mae = np.mean(np.abs(true_energies - pred_energies))
    # print("Energy MAE:", mae)
    message = "{} energy MAE: {}\t time: {}\n".format(data_type, mae, time_took)
    log(log_filename, message)
​
    list_of_error_per_atom = []
    with open('{}_prediction_result.csv'.format(data_type), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i, image in enumerate(data_list):
            num_atoms = len(image.get_atomic_numbers())
            error = pred_energies[i] - true_energies[i]
            per_atom_error = error / num_atoms
            list_of_error_per_atom.append(per_atom_error)
            writer.writerow([i, num_atoms,true_energies[i], pred_energies[i], error, per_atom_error, abs(error), abs(per_atom_error)])
​
​
def train_model(train_list, test_list, num_gaussian, max_MCSH_order, trial_num, log_filename):
​
    MCSHs = construct_parameter_set(num_gaussian, max_MCSH_order, log_filename = log_filename)

    elements = ["H","O","C"]
    
    config = {
        "model": {"name":"singlenn",
                  "get_forces": False, 
                  "num_layers": 3, 
                  "num_nodes": 50, 
                  #"elementwise":False, 
                  "batchnorm": True},
        "optim": {
            "gpus":0,
            #"force_coefficient": 0.04,
            "force_coefficient": 0.0,
            "lr": 5e-3,
            "batch_size": 128,
            "epochs": 4000,
            "loss": "mae",
            "scheduler": {"policy": "StepLR", "params": {"step_size": 1000, "gamma": 0.5}},
        },
        "dataset": {
            "raw_data": train_list,
            "val_split": 0.2,
            "elements": elements,
            "fp_scheme": "gmpordernorm",
            "fp_params": MCSHs,
            "save_fps": False,
            "scaling": {"type": "normalize", "range": (-1.0, 1.0),"elementwise":False}
        },
        "cmd": {
            "debug": False,
            "run_dir": "./",
            "seed": trial_num,
            "identifier": "test",
            "verbose": True,
            "logger": False,
        },
    }
​
​
    trainer = AtomsTrainer(config)
    trainer.train()
​
    test_model(trainer, train_list, data_type = "train", log_filename = log_filename)
    test_model(trainer, test_list, data_type = "test", log_filename = log_filename)
​
​
    return
​
def load_data(systems, trial_num):
    train_data = []
    test_data  = []
    for system in systems:
        train_filename = "../data/{}_train_data_{}.p".format(system, trial_num)
        test_filename = "../data/{}_test_data_{}.p".format(system, trial_num)
        train_data += pickle.load( open( train_filename, "rb" ) )
        test_data  += pickle.load( open( test_filename, "rb" ) )
    return train_data, test_data
​
torch.set_num_threads(1)
trial_num = int(sys.argv[1])
​
log_filename = "info.log"
​
systems = ["aspirin"]
​
​
​
cwd = os.getcwd()
​
num_gaussian = int(sys.argv[2])
max_MCSH_order = int(sys.argv[3])
​
​
folder_name = "ngaussian_{}_MCSH_{}_ordernorm_trial_{}_singleNN".format(num_gaussian, max_MCSH_order, trial_num)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
os.chdir(folder_name)
train_data, test_data = load_data(systems, trial_num)
​
log(log_filename, str(systems) + "\n")
train_model(train_data, test_data, num_gaussian, max_MCSH_order, trial_num, log_filename)
​
os.chdir(cwd)