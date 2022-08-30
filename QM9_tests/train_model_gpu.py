# importing the module 
import json 
import ase
import numpy as np
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from amptorch.trainer import AtomsTrainer
import pickle
import random
import math
import sys
import torch
import os
import csv
import time
​
def log(log_filename, message):
    f = open(log_filename, "a")
    f.write(message)
    f.close()
    return
​
​
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
def load_linear_fit_result(linear_fit_result_filename):
​
    correction_dict = {}
    with open(linear_fit_result_filename) as fp: 
        Lines = fp.readlines() 
        for line in Lines: 
            temp = line.split()
            print(line.split())
            correction_dict[temp[0]]=float(temp[1])
    return correction_dict
​
def predict_data(trainer, test_images, folder_name, linear_fit_result_filename = "../linear_model_result.dat", image_type = "test"):
    cwd = os.getcwd()
    os.chdir(folder_name)
    linear_fit_result = load_linear_fit_result(linear_fit_result_filename)
    os.chdir(cwd)
    
    predictions = trainer.predict(test_images, disable_tqdm = False)
    true_energies = np.array([image.get_potential_energy() for image in test_images])
    pred_energies = np.array(predictions["energy"])
    print(true_energies.shape)
    print(pred_energies.shape)
​
    pickle.dump( true_energies, open( "{}_true_energies.p".format(image_type), "wb" ) )
    pickle.dump( pred_energies, open( "{}_pred_energies.p".format(image_type), "wb" ) )
​
    mae_result = np.mean(np.abs(true_energies - pred_energies))
    print("Energy MAE:", mae_result)
    os.chdir(folder_name)
    list_of_error_per_atom = []
    with open('{}_prediction_result.csv'.format(image_type), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i, image in enumerate(test_images):
            num_atoms = len(image.get_atomic_numbers())
            total_energy_pred = pred_energies[i]
            total_energy_true = true_energies[i]
            for symbol in image.get_chemical_symbols():
                total_energy_pred += linear_fit_result[symbol]
                total_energy_true += linear_fit_result[symbol]
            
            error = pred_energies[i] - true_energies[i]
            per_atom_error = error / num_atoms
            list_of_error_per_atom.append(per_atom_error)
            writer.writerow([i, num_atoms,true_energies[i], pred_energies[i], total_energy_true, total_energy_pred,
                             error, per_atom_error, abs(error), abs(per_atom_error)])
    os.chdir(cwd)
    return mae_result
​
​
​
dataset_name = sys.argv[1]
nsigmas = int(sys.argv[2])
MCSHs_index = int(sys.argv[3])
NN_index = int(sys.argv[4])
​
​
num_gpu = torch.cuda.device_count()
print("****\n Found {} GPUs \n****\n\n".format(num_gpu))
​
train_filename = "../train.p"
test_filename = "../test.p"
cwd = os.getcwd()
folder_name = "./trial_{}/test_GELU_ordernorm_sigma{}linwide_MCSH{}_NN{}_sqrt".format(dataset_name, nsigmas, MCSHs_index,NN_index)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
os.chdir(folder_name)
train_images, test_images = load_training_data(train_filename, test_filename)

sigmas = np.linspace(0.0,2.0,nsigmas+1,endpoint=True)[1:].tolist()
print(sigmas)
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
MCSHs = MCSHs_dict[MCSHs_index]
​
​
GMP = {   "MCSHs": MCSHs,
            "atom_gaussians": {
                        "H": "../../psp_pseudo_v3/H_pseudodensity.g",
                        "C": "../../psp_pseudo_v3/C_pseudodensity.g",
                        "O": "../../psp_pseudo_v3/O_pseudodensity.g",
                        "N": "../../psp_pseudo_v3/N_pseudodensity.g",
                        "F": "../../psp_pseudo_v3/F_pseudodensity.g",
                  },
            # "cutoff": 10.0,
            "square":False,
            "solid_harmonics": True,
}
​
​
elements = ["H","C","N","O","F"]
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
​
config = {
    "model": {"name":"singlenn",
                  "get_forces": False, 
                  "hidden_layers": hidden_layers, 
                  "elementwise":False,
                  "activation":torch.nn.GELU,
                  "batchnorm": True},
    "optim": {
            "gpus":num_gpu,
            "force_coefficient": 0.0,
            "lr": 5e-3,
            "batch_size": 128,
            "epochs": 4000,
            "loss": "mae",
            "scheduler": {"policy": "StepLR", "params": {"step_size": 1000, "gamma": 0.5}}
    },
    "dataset": {
            "raw_data": train_images,
            "val_split": 0.1,
            "elements": elements,
            "fp_scheme": "gmpordernorm",
            "fp_params": GMP,
            "save_fps": True,
            "scaling": {"type": "normalize", "range": (-1, 1),"elementwise":False,"threshold": 1e-8}
        },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "data{}-GELU_ordernorm_sigma{}lin_MCSH{}_NN{}".format(dataset_name, nsigmas, MCSHs_index,NN_index),
        "verbose": True,
        "logger": False,
        "dtype": torch.DoubleTensor
    },
}
​
​
​
trainer = AtomsTrainer(config)
print("training")
os.chdir(cwd)
tr_start = time.time()
trainer.train()
training_time = time.time() - tr_start
​
print("end training")
​
pr_start = time.time()
train_mae = predict_data(trainer, train_images, folder_name, image_type = "train")
test_mae = predict_data(trainer, test_images, folder_name, image_type = "test")
predict_time = time.time() - pr_start
​
os.chdir(folder_name)
message = "ordernorm sqrt 1e-8scale\t{}linwide\t{}\t{}\tGELU\t{}\t{}\ttraining:{}\tpred:{}\n".format(
                    nsigmas, 
                    MCSHs_index, 
                    hidden_layers,
                    train_mae,
                    test_mae,
                    training_time,
                    predict_time
                    )
log("../pseudo_train_result.dat",message)
os.chdir(cwd)