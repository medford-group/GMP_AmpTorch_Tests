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

def log(log_filename, message):
    f = open(log_filename, "a")
    f.write(message)
    f.close()
    return



def load_training_data(training_filename, test_filename):


    training_list = pickle.load( open( training_filename, "rb" ) )
    test_list = pickle.load( open( test_filename, "rb" ) )

    return training_list, test_list


def load_linear_fit_result(linear_fit_result_filename):

    correction_dict = {}
    with open(linear_fit_result_filename) as fp: 
        Lines = fp.readlines() 
        for line in Lines: 
            temp = line.split()
            print(line.split())
            correction_dict[temp[0]]=float(temp[1])
#     print(correction_dict)        
    return correction_dict

def predict_data(trainer, test_images, linear_fit_result_filename = "../linear_model_result.dat", image_type = "test"):
    linear_fit_result = load_linear_fit_result(linear_fit_result_filename)
    
    predictions = trainer.predict(test_images)
    true_energies = np.array([image.get_potential_energy() for image in test_images])
    pred_energies = np.array(predictions["energy"])
    print(true_energies.shape)
    print(pred_energies.shape)

    pickle.dump( true_energies, open( "{}_true_energies.p".format(image_type), "wb" ) )
    pickle.dump( pred_energies, open( "{}_pred_energies.p".format(image_type), "wb" ) )

    mae_result = np.mean(np.abs(true_energies - pred_energies))
    print("Energy MAE:", mae_result)

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

    return mae_result


#torch.set_num_threads(1)
trail_num = sys.argv[1]
checkpoint_name = sys.argv[2]
sigmas_index = int(sys.argv[3])
MCSHs_index = int(sys.argv[4])
num_nodes = int(sys.argv[5])
num_layers = int(sys.argv[6])
batch_size = int(sys.argv[7])
cutoff_distance = float(sys.argv[8])

train_filename = "../QM9_train_{}.p".format(trail_num)
test_filename = "../QM9_test_{}.p".format(trail_num)
folder_name = "./trial_{}/test_sigma{}_MCSH{}_nodes{}_layers{}_batch{}_cutoff{}".format(trail_num, sigmas_index, MCSHs_index, num_nodes,num_layers,batch_size,cutoff_distance)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
os.chdir(folder_name)
train_images, test_images = load_training_data(train_filename, test_filename)

sigmas_dict = {
    37: [0.02,0.05,0.08,0.12,0.16,0.2,0.24,0.28,0.32,0.36,0.4,0.45,0.5,0.56,0.62,0.69,0.76,0.84,0.92,1.01,1.1,1.2,1.3,1.4,1.52,1.66,1.82,2.0,2.42,2.66,2.92,3.2,3.5,3.9,4.4,5.0],
    19: [0.02,0.08,0.16,0.24,0.32,0.4,0.5,0.62,0.76,0.92,1.1,1.3,1.52,1.82,2.2,2.66,3.2,3.9,5.0],
    13: [0.02,0.12,0.24,0.36,0.5,0.69,0.92,1.2,1.52,2.0,2.66,3.5,5.0],
    10: [0.02,0.16,0.32,0.5,0.76,1.1,1.52,2.2,3.2,5.0],
    8: [0.02,0.2,0.4,0.69,1.1,1.66,2.66,4.4],

}

sigmas = sigmas_dict[sigmas_index]

MCSHs_dict = {
    1: {   
                "0": {"groups": [1], "sigmas": sigmas},
                "1": {"groups": [1], "sigmas": sigmas},
          },
    2: {   
                "0": {"groups": [1], "sigmas": sigmas},
                "1": {"groups": [1], "sigmas": sigmas},
                "2": {"groups": [1,2], "sigmas": sigmas},
          },
    3: {   
                "0": {"groups": [1], "sigmas": sigmas},
                "1": {"groups": [1], "sigmas": sigmas},
                "2": {"groups": [1,2], "sigmas": sigmas},
                "3": {"groups": [1,2,3], "sigmas": sigmas},
          },
    4: {   
                "0": {"groups": [1], "sigmas": sigmas},
                "1": {"groups": [1], "sigmas": sigmas},
                "2": {"groups": [1,2], "sigmas": sigmas},
                "3": {"groups": [1,2,3], "sigmas": sigmas},
                "4": {"groups": [1,2,3,4], "sigmas": sigmas},
          },
    5: {   
                "0": {"groups": [1], "sigmas": sigmas},
                "1": {"groups": [1], "sigmas": sigmas},
                "2": {"groups": [1,2], "sigmas": sigmas},
                "3": {"groups": [1,2,3], "sigmas": sigmas},
                "4": {"groups": [1,2,3,4], "sigmas": sigmas},
                "5": {"groups": [1,2,3,4,5], "sigmas": sigmas},
          },
}

MCSHs = MCSHs_dict[MCSHs_index]


MCSHs = {   "MCSHs": MCSHs,
            "atom_gaussians": {
                        "H": "../../valence_gaussians/H_pseudodensity_2.g",
                        "C": "../../valence_gaussians/C_pseudodensity_4.g",
                        "O": "../../valence_gaussians/O_pseudodensity_4.g",
                        "N": "../../valence_gaussians/N_pseudodensity_4.g",
                        "F": "../../valence_gaussians/F_pseudodensity_4.g",
                  },
            "cutoff": cutoff_distance
}



# elements = ["Cu", "C", "O"]
elements = ["H","C","N","O","F"]
config = {
    "model": {"name":"singlenn",
                  "get_forces": False, 
                  "num_layers": num_layers, 
                  "num_nodes": num_nodes, 
                  #"elementwise":False,
                  #"activation":torch.nn.ReLU,
                  "batchnorm": True,
                  "dropout":False,
                  "dropout_rate": 0.2},
    "optim": {
            "gpus":1,
            #"force_coefficient": 0.04,
            "force_coefficient": 0.0,
            #"optimizer": torch.optim.SGD,
            "lr": 1e-2,
            "batch_size": batch_size,
            "epochs": 12000,
            "loss": "mae",
            "scheduler": {"policy": "StepLR", "params": {"step_size": 2000, "gamma": 0.5}}
    },
    "dataset": {
            "raw_data": train_images,
            "val_split": 0.1,
            "elements": elements,
            "fp_scheme": "mcsh",
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



trainer = AtomsTrainer(config)
#trainer.load_pretrained(checkpoint_name)
print("training")
trainer.train()
print("end training")

train_mae = predict_data(trainer, train_images, image_type = "train")
test_mae = predict_data(trainer, test_images, image_type = "test")

message = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    sigmas_index, 
                    MCSHs_index, 
                    num_nodes,
                    num_layers,
                    batch_size,
                    cutoff_distance,
                    train_mae,
                    test_mae
                    )
log("../pseudo_train_result.dat",message)

