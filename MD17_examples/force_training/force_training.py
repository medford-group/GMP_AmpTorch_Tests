# importing the module 
import json 
import ase
import numpy as np
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from amptorch.trainer import AtomsTrainer
import amptorch
import pickle
import random
import math
import sys
import torch
import os
import csv
​
def log(log_filename, message):
    f = open(log_filename, "a")
    f.write(message)
    f.close()
    return
​
​
def load_data_to_atoms(system_data):
    atoms_list = []
    for i, atom in enumerate(system_data["atoms"]):
        atoms_list.append(Atom(atom, np.array(system_data["coordinates"][i])))
    system = Atoms(atoms_list, cell=[50, 50, 50])
    system.center()
    calc = SinglePointCalculator(system,energy = system_data['u0'])
#     calc = EMT(energy = system_data['u0'])
    system.set_calculator(calc)
    # print(system.get_potential_energy())
    # print(system_data['u0'])
    return system
​
def load_training_data(training_filename, test_filename):
​
​
    training_list = pickle.load( open( training_filename, "rb" ) )
    test_list = pickle.load( open( test_filename, "rb" ) )
​
    return training_list, test_list
​
​
def predict_data(trainer, test_images, linear_fit_result_filename = "../linear_model_result.dat", image_type = "test"):
    
    predictions = trainer.predict(test_images)
    true_energies = np.array([image.get_potential_energy() for image in test_images])
    true_forces = np.array([image.get_forces() for image in test_images])
    pred_energies = np.array(predictions["energy"])
    pred_forces = np.array(predictions["forces"])
    #print(true_energies.shape)
    #print(pred_energies.shape)
​
    pickle.dump( true_energies, open( "{}_true_energies.p".format(image_type), "wb" ) )
    pickle.dump( pred_energies, open( "{}_pred_energies.p".format(image_type), "wb" ) )
​
    mae_result = np.mean(np.abs(true_energies - pred_energies))
    force_mae_result = np.mean(np.abs(true_forces - pred_forces))
    print("Energy MAE:", mae_result)
​
    list_of_error_per_atom = []
    with open('{}_prediction_result.csv'.format(image_type), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i, image in enumerate(test_images):
            num_atoms = len(image.get_atomic_numbers())
            total_energy_pred = pred_energies[i]
            total_energy_true = true_energies[i]
            
            error = pred_energies[i] - true_energies[i]
            per_atom_error = error / num_atoms
            list_of_error_per_atom.append(per_atom_error)
            writer.writerow([i, num_atoms,true_energies[i], pred_energies[i], total_energy_true, total_energy_pred,
                             error, per_atom_error, abs(error), abs(per_atom_error), force_mae_result])
​
    return mae_result, force_mae_result
​
print(amptorch.__file__)
#torch.set_num_threads(1)
trail_num = sys.argv[1]
num_gaussian = int(sys.argv[2])
max_MCSH_order = int(sys.argv[3])
num_nodes = int(sys.argv[4])
num_layers = int(sys.argv[5])
batch_size = int(sys.argv[6])
force_coef = float(str(sys.argv[7]))
​
train_filename = "../train_data.p".format(trail_num)
test_filename = "../test_data.p".format(trail_num)
folder_name = "./aspirin_trial_{}/test_sigma{}_MCSH{}_nodes{}_layers{}_batch{}_force{}".format(trail_num, num_gaussian, max_MCSH_order, num_nodes,num_layers,batch_size,force_coef)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
os.chdir(folder_name)
train_images, test_images = load_training_data(train_filename, test_filename)
​
sigmas = np.linspace(0.0,2.0,num_gaussian+1,endpoint=True)[1:].tolist()
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
​
​
elements = ["H","O","C"]
config = {
    "model": {"name":"singlenn",
                  "get_forces": True, 
                  "num_layers": num_layers, 
                  "num_nodes": num_nodes, 
                  "elementwise":False,
                  #"activation":torch.nn.ReLU,
                  "batchnorm": True
          },
    "optim": {
            "gpus":0,
            #"force_coefficient": 0.04,
            "force_coefficient": force_coef,
            #"optimizer": torch.optim.SGD,
            "lr": 5e-3,
            "batch_size": 128,
            "epochs": 4000,
            "loss": "mae",
            "scheduler": {"policy": "StepLR", "params": {"step_size": 1000, "gamma": 0.5}},
    },
    "dataset": {
            "raw_data": train_images,
            "val_split": 0.2,
            "elements": elements,
            "fp_scheme": "gmpordernorm",
            "fp_params": MCSHs,
            "save_fps": False,
            "scaling": {"type": "normalize", "range": (-1, 1),"elementwise":False}
        },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "test",
        "verbose": True,
        "logger": False,
    },
}
​
​
​
trainer = AtomsTrainer(config)
print("training")
trainer.train()
print("end training")
​
train_mae, train_force_mae = predict_data(trainer, train_images, image_type = "train")
test_mae, test_force_mae = predict_data(trainer, test_images, image_type = "test")
​
message = "Ordernorm {}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    num_gaussian, 
                    max_MCSH_order, 
                    num_nodes,
                    num_layers,
                    batch_size,
                    force_coef,
                    train_mae,
                    test_mae,
                    train_force_mae,
                    test_force_mae
                    )
log("../pseudo_train_result.dat",message)
​