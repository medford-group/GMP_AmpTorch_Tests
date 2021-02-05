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



def load_linear_fit_result(linear_fit_result_filename):

    correction_dict = {}
    with open(linear_fit_result_filename) as fp: 
        Lines = fp.readlines() 
        for line in Lines: 
            temp = line.split()
            print(line.split())
            correction_dict[temp[0]]=float(temp[1])      
    return correction_dict

def predict_data(trainer, test_images, linear_fit_result_filename = "./linear_model_result.dat", image_type = "test"):
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


torch.set_num_threads(1)
trail_num = sys.argv[1]
checkpoint_name = sys.argv[2]

sigmas_index = int(sys.argv[3])
MCSHs_index = int(sys.argv[4])
num_nodes = int(sys.argv[5])
num_layers = int(sys.argv[6])
batch_size = int(sys.argv[7])
cutoff_distance = float(sys.argv[8])
num_training = int(sys.argv[9])

train_filename = "OCP_train_s2ef_train_200K.p"
folder_name = "trial_{}".format(trail_num)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
os.chdir(folder_name)

train_images = pickle.load( open( training_filename, "rb" ) )

potential_files = {
"H":	"../valence_gaussians/H_pseudodensity_2.g",
"Li":	"../valence_gaussians/Li_pseudodensity_2.g",
"Be":	"../valence_gaussians/Be_pseudodensity_2.g",
"B":	"../valence_gaussians/B_pseudodensity_3.g",
"C":	"../valence_gaussians/C_pseudodensity_4.g",
"N":	"../valence_gaussians/N_pseudodensity_4.g",
"O":	"../valence_gaussians/O_pseudodensity_4.g",
"F":	"../valence_gaussians/F_pseudodensity_4.g",
"Na":	"../valence_gaussians/Na_pseudodensity_4.g",
"Mg":	"../valence_gaussians/Mg_pseudodensity_4.g",
"Al":	"../valence_gaussians/Al_pseudodensity_4.g",
"Si":	"../valence_gaussians/Si_pseudodensity_5.g",
"P":	"../valence_gaussians/P_pseudodensity_5.g",
"S":	"../valence_gaussians/S_pseudodensity_5.g",
"Cl":	"../valence_gaussians/Cl_pseudodensity_5.g",
"K":	"../valence_gaussians/K_pseudodensity_4.g",
"Ca":	"../valence_gaussians/Ca_pseudodensity_4.g",
"Sc":	"../valence_gaussians/Sc_pseudodensity_5.g",
"Ti":	"../valence_gaussians/Ti_pseudodensity_5.g",
"V":	"../valence_gaussians/V_pseudodensity_4.g",
"Cr":	"../valence_gaussians/Cr_pseudodensity_4.g",
"Mn":	"../valence_gaussians/Mn_pseudodensity_4.g",
"Fe":	"../valence_gaussians/Fe_pseudodensity_4.g",
"Co":	"../valence_gaussians/Co_pseudodensity_4.g",
"Ni":	"../valence_gaussians/Ni_pseudodensity_4.g",
"Cu":	"../valence_gaussians/Cu_pseudodensity_4.g",
"Zn":	"../valence_gaussians/Zn_pseudodensity_4.g",
"Ga":	"../valence_gaussians/Ga_pseudodensity_5.g",
"Ge":	"../valence_gaussians/Ge_pseudodensity_5.g",
"As":	"../valence_gaussians/As_pseudodensity_5.g",
"Se":	"../valence_gaussians/Se_pseudodensity_5.g",
"Br":	"../valence_gaussians/Br_pseudodensity_5.g",
"Rb":	"../valence_gaussians/Rb_pseudodensity_5.g",
"Sr":	"../valence_gaussians/Sr_pseudodensity_5.g",
"Y":	"../valence_gaussians/Y_pseudodensity_5.g",
"Zr":	"../valence_gaussians/Zr_pseudodensity_4.g",
"Nb":	"../valence_gaussians/Nb_pseudodensity_4.g",
"Mo":	"../valence_gaussians/Mo_pseudodensity_4.g",
"Tc":	"../valence_gaussians/Tc_pseudodensity_4.g",
"Ru":	"../valence_gaussians/Ru_pseudodensity_4.g",
"Rh":	"../valence_gaussians/Rh_pseudodensity_4.g",
"Pd":	"../valence_gaussians/Pd_pseudodensity_4.g",
"Ag":	"../valence_gaussians/Ag_pseudodensity_4.g",
"Cd":	"../valence_gaussians/Cd_pseudodensity_4.g",
"In":	"../valence_gaussians/In_pseudodensity_4.g",
"Sn":	"../valence_gaussians/Sn_pseudodensity_4.g",
"Sb":	"../valence_gaussians/Sb_pseudodensity_4.g",
"Te":	"../valence_gaussians/Te_pseudodensity_4.g",
"I":	"../valence_gaussians/I_pseudodensity_4.g",
"Cs":	"../valence_gaussians/Cs_pseudodensity_5.g",
"Ba":	"../valence_gaussians/Ba_pseudodensity_5.g",
"Hf":	"../valence_gaussians/Hf_pseudodensity_5.g",
"Ta":	"../valence_gaussians/Ta_pseudodensity_5.g",
"W":	"../valence_gaussians/W_pseudodensity_7.g",
"Re":	"../valence_gaussians/Re_pseudodensity_6.g",
"Os":	"../valence_gaussians/Os_pseudodensity_6.g",
"Ir":	"../valence_gaussians/Ir_pseudodensity_6.g",
"Pt":	"../valence_gaussians/Pt_pseudodensity_6.g",
"Au":	"../valence_gaussians/Au_pseudodensity_6.g",
"Hg":	"../valence_gaussians/Hg_pseudodensity_6.g",
"Tl":	"../valence_gaussians/Tl_pseudodensity_6.g",
"Pb":	"../valence_gaussians/Pb_pseudodensity_6.g",
"Bi":	"../valence_gaussians/Bi_pseudodensity_6.g",}
sigmas_dict = {
    37: [0.02,0.05,0.08,0.12,0.16,0.2,0.24,0.28,0.32,0.36,0.4,0.45,0.5,0.56,0.62,0.69,0.76,0.84,0.92,1.01,1.1,1.2,1.3,1.4,1.52,1.66,1.82,2.0,2.42,2.66,2.92,3.2,3.5,3.9,4.4,5.0],
    19: [0.02,0.08,0.16,0.24,0.32,0.4,0.5,0.62,0.76,0.92,1.1,1.3,1.52,1.82,2.2,2.66,3.2,3.9,5.0],
    13: [0.02,0.12,0.24,0.36,0.5,0.69,0.92,1.2,1.52,2.0,2.66,3.5,5.0],
    10: [0.02,0.16,0.32,0.5,0.76,1.1,1.52,2.2,3.2,5.0],
    8: [0.02,0.2,0.4,0.69,1.1,1.66,2.66,4.4],

}

sigmas = sigmas_dict[sigmas_index]

MCSHs_dict = {
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
            "atom_gaussians": potential_files,
            "cutoff": cutoff_distance
}



elements = ['Pt',
 'Al',
 'V',
 'Pd',
 'Fe',
 'Sn',
 'Ge',
 'Bi',
 'Ir',
 'Re',
 'Cd',
 'Cr',
 'Ag',
 'Hf',
 'Ru',
 'Ti',
 'Cs',
 'Os',
 'N',
 'As',
 'O',
 'S',
 'Mo',
 'Ta',
 'Zn',
 'Y',
 'Mn',
 'Na',
 'Rh',
 'Hg',
 'C',
 'Co',
 'Nb',
 'Sc',
 'Sr',
 'H',
 'Au',
 'Ga',
 'Tl',
 'K',
 'Se',
 'B',
 'Pb',
 'Ca',
 'Cl',
 'Cu',
 'Zr',
 'Rb',
 'P',
 'W',
 'Tc',
 'Te',
 'Ni',
 'Sb',
 'Si',
 'In']
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
        "lr": 5e-3,
        "batch_size": batch_size,
        "epochs":7500,
        "loss": "mae",
        "scheduler": {"policy": "StepLR", "params": {"step_size": 1500, "gamma": 0.5}}
    },
    "dataset": {
        "raw_data": train_images[:num_training],
        "val_split": 0.1,
        "elements": elements,
        "fp_scheme": "mcsh",
        "fp_params": MCSHs,
        "save_fps": True,
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


train_mae = predict_data(trainer, train_images[:num_training], image_type = "train")
test_mae = predict_data(trainer, train_images[num_training:num_training+25000], image_type = "test")

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
log("./train_result.dat",message)
