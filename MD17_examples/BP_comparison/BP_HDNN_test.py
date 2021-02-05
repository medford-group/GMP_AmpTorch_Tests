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
from amptorch.dataset import AtomsDataset
import csv
from skorch.callbacks import LRScheduler


def log(log_filename, message):
    f = open(log_filename, "a")
    f.write(message)
    f.close()
    return

def construct_parameter_set(descriptor_set, log_filename = "info.log"):
    descriptor_set_dict = {
    1: {
        "default": {
            "G2": {
                "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=8).tolist(),
                "rs_s": [0,1.5],
            },
            "G4": {"etas": [0.001, 0.01, 0.03], "zetas": [1.0, 2.0, 4.0], "gammas": [1.0, -1.0]},
            "cutoff": 10.0,
            }
        },

    2: {
            "default": {
                "G2": {
                    "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=8).tolist(),
                    "rs_s": [0,1.5],
                },
                "G4": {"etas": [0.01, 0.03], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
                "cutoff": 10.0,
            },
        },

    3: {
            "default": {
                "G2": {
                    "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=8).tolist(),
                    "rs_s": [0],
                },
                "G4": {"etas": [0.01], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
                "cutoff": 10.0,
            },
        },

    4: {
        "default": {
            "G2": {
                "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=8).tolist(),
                "rs_s": [0],
            },
            "G4": {"etas": [0.001, 0.01, 0.03], "zetas": [1.0, 2.0, 4.0], "gammas": [1.0, -1.0]},
            "cutoff": 10.0,
            }
        },

    5: {
            "default": {
                "G2": {
                    "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=8).tolist(),
                    "rs_s": [0],
                },
                "G4": {"etas": [0.01, 0.03], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
                "cutoff": 10.0,
            },
        },

    6: {
            "default": {
                "G2": {
                    "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=8).tolist(),
                    "rs_s": [0,1.5],
                },
                "G4": {"etas": [0.01], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
                "cutoff": 10.0,
            },
        },

    7: {
        "default": {
            "G2": {
                "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4).tolist(),
                "rs_s": [0,1.5],
            },
            "G4": {"etas": [0.001, 0.01, 0.03], "zetas": [1.0, 2.0, 4.0], "gammas": [1.0, -1.0]},
            "cutoff": 10.0,
            }
        },

    8: {
            "default": {
                "G2": {
                    "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4).tolist(),
                    "rs_s": [0,1.5],
                },
                "G4": {"etas": [0.01, 0.03], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
                "cutoff": 10.0,
            },
        },

    9: {
            "default": {
                "G2": {
                    "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4).tolist(),
                    "rs_s": [0],
                },
                "G4": {"etas": [0.01], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
                "cutoff": 10.0,
            },
        },

    10: {
        "default": {
            "G2": {
                "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4).tolist(),
                "rs_s": [0],
            },
            "G4": {"etas": [0.001, 0.01, 0.03], "zetas": [1.0, 2.0, 4.0], "gammas": [1.0, -1.0]},
            "cutoff": 10.0,
            }
        },

    11: {
            "default": {
                "G2": {
                    "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4).tolist(),
                    "rs_s": [0],
                },
                "G4": {"etas": [0.01, 0.03], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
                "cutoff": 10.0,
            },
        },

    12: {
            "default": {
                "G2": {
                    "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4).tolist(),
                    "rs_s": [0,1.5],
                },
                "G4": {"etas": [0.01], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
                "cutoff": 10.0,
            },
        }


    }

    Gs = descriptor_set_dict[descriptor_set]
    log(log_filename, json.dumps(Gs,indent = 4) + "\n")
    return Gs

def test_model(trainer, data_list, data_type = "test", log_filename = "info.log"):
    start = time.time()
    predictions = trainer.predict(data_list)
    time_took = time.time() - start

    true_energies = np.array([image.get_potential_energy() for image in data_list])
    pred_energies = np.array(predictions["energy"])
    mae = np.mean(np.abs(true_energies - pred_energies))
    # print("Energy MAE:", mae)
    message = "{} energy MAE: {}\t time: {}\n".format(data_type, mae, time_took)
    log(log_filename, message)

    list_of_error_per_atom = []
    with open('{}_prediction_result.csv'.format(data_type), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i, image in enumerate(data_list):
            num_atoms = len(image.get_atomic_numbers())
            error = pred_energies[i] - true_energies[i]
            per_atom_error = error / num_atoms
            list_of_error_per_atom.append(per_atom_error)
            writer.writerow([i, num_atoms,true_energies[i], pred_energies[i], error, per_atom_error, abs(error), abs(per_atom_error)])


def train_model(train_list, test_list, descriptor_set, trial_num, log_filename):

    Gs = construct_parameter_set(descriptor_set, log_filename = log_filename)


    # elements = ["Cu", "C", "O"]
    elements = ["H","O","C"]


    config = {
        "model": {"name":"bpnn",
                  "get_forces": False, 
                  "num_layers": 3, 
                  "num_nodes": 50, 
                  #"elementwise":False, 
                  "batchnorm": True},

        "optim": {
            "gpus":0,
            #"force_coefficient": 0.04,
            "force_coefficient": 0.0,
            "lr": 1e-2,
            "batch_size": 32,
            "epochs": 6000,
            "loss": "mae",
            "scheduler": {"policy": "StepLR", "params": {"step_size": 1000, "gamma": 0.5}},
        },
        "dataset": {
            "raw_data": train_list,
            "val_split": 0.2,
            "elements": elements,
            "fp_scheme": "gaussian",
            "fp_params": Gs,
            "save_fps": False,
            "scaling": {"type": "normalize", "range": (0, 1)}
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


    trainer = AtomsTrainer(config)
    trainer.train()

    test_model(trainer, train_list, data_type = "train", log_filename = log_filename)
    test_model(trainer, test_list, data_type = "test", log_filename = log_filename)


    return

def load_data(systems, trial_num):
    train_data = []
    test_data  = []
    for system in systems:
        train_filename = "../data/{}_train_data_{}.p".format(system, trial_num)
        test_filename = "../data/{}_test_data_{}.p".format(system, trial_num)
        train_data += pickle.load( open( train_filename, "rb" ) )
        test_data  += pickle.load( open( test_filename, "rb" ) )
    return train_data, test_data

torch.set_num_threads(1)
trial_num = int(sys.argv[1])
descriptor_set = int(sys.argv[2])

log_filename = "info.log"

systems = ["aspirin"]



cwd = os.getcwd()


folder_name = "BP_set_{}_trial_{}".format(descriptor_set, trial_num)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
os.chdir(folder_name)
train_data, test_data = load_data(systems, trial_num)

log(log_filename, str(systems) + "\n")
train_model(train_data, test_data, descriptor_set, trial_num, log_filename)

os.chdir(cwd)

