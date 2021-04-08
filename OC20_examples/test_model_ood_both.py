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
#     print(correction_dict)        
    return correction_dict

def predict_data(trainer, test_images, set_index, result_dirname, linear_fit_result_filename = "./linear_model_result.dat", image_type = "test"):
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

    if not os.path.exists(result_dirname):
        os.makedirs(result_dirname)


    list_of_error_per_atom = []
    with open(result_dirname + '/{}_prediction_result_id.csv'.format(image_type), 'a', newline='') as csvfile:
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
            writer.writerow([set_index, i, num_atoms,true_energies[i], pred_energies[i], total_energy_true, total_energy_pred,
                             error, per_atom_error, abs(error), abs(per_atom_error)])

    log(result_dirname + '/test_mae_ood_both.dat', '{}\t{}\n'.format(set_index,mae_result))
    return mae_result


torch.set_num_threads(1)
trail_num = "data_full_20M"
checkpoint_name = sys.argv[1]

sigmas_index = int(sys.argv[2])
MCSHs_index = int(sys.argv[3])
num_nodes = int(sys.argv[4])
num_layers = int(sys.argv[5])
num_training = int(sys.argv[6])
cutoff_distance = float(sys.argv[7])

set_index = int(sys.argv[8])

test_filename = "s2ef_val_ood_both_images/{}.p".format(set_index)
folder_name = "trial_{}".format(trail_num)

os.chdir(folder_name)


test_images = pickle.load( open( test_filename, "rb" ) )


potential_files = {
"H":	"../MCSH_potential_pseudo2/H_pseudodensity_2.g",
"Li":	"../MCSH_potential_pseudo2/Li_pseudodensity_2.g",
"Be":	"../MCSH_potential_pseudo2/Be_pseudodensity_2.g",
"B":	"../MCSH_potential_pseudo2/B_pseudodensity_3.g",
"C":	"../MCSH_potential_pseudo2/C_pseudodensity_4.g",
"N":	"../MCSH_potential_pseudo2/N_pseudodensity_4.g",
"O":	"../MCSH_potential_pseudo2/O_pseudodensity_4.g",
"F":	"../MCSH_potential_pseudo2/F_pseudodensity_4.g",
"Na":	"../MCSH_potential_pseudo2/Na_pseudodensity_4.g",
"Mg":	"../MCSH_potential_pseudo2/Mg_pseudodensity_4.g",
"Al":	"../MCSH_potential_pseudo2/Al_pseudodensity_4.g",
"Si":	"../MCSH_potential_pseudo2/Si_pseudodensity_5.g",
"P":	"../MCSH_potential_pseudo2/P_pseudodensity_5.g",
"S":	"../MCSH_potential_pseudo2/S_pseudodensity_5.g",
"Cl":	"../MCSH_potential_pseudo2/Cl_pseudodensity_5.g",
"K":	"../MCSH_potential_pseudo2/K_pseudodensity_4.g",
"Ca":	"../MCSH_potential_pseudo2/Ca_pseudodensity_4.g",
"Sc":	"../MCSH_potential_pseudo2/Sc_pseudodensity_5.g",
"Ti":	"../MCSH_potential_pseudo2/Ti_pseudodensity_5.g",
"V":	"../MCSH_potential_pseudo2/V_pseudodensity_4.g",
"Cr":	"../MCSH_potential_pseudo2/Cr_pseudodensity_4.g",
"Mn":	"../MCSH_potential_pseudo2/Mn_pseudodensity_4.g",
"Fe":	"../MCSH_potential_pseudo2/Fe_pseudodensity_4.g",
"Co":	"../MCSH_potential_pseudo2/Co_pseudodensity_4.g",
"Ni":	"../MCSH_potential_pseudo2/Ni_pseudodensity_4.g",
"Cu":	"../MCSH_potential_pseudo2/Cu_pseudodensity_4.g",
"Zn":	"../MCSH_potential_pseudo2/Zn_pseudodensity_4.g",
"Ga":	"../MCSH_potential_pseudo2/Ga_pseudodensity_5.g",
"Ge":	"../MCSH_potential_pseudo2/Ge_pseudodensity_5.g",
"As":	"../MCSH_potential_pseudo2/As_pseudodensity_5.g",
"Se":	"../MCSH_potential_pseudo2/Se_pseudodensity_5.g",
"Br":	"../MCSH_potential_pseudo2/Br_pseudodensity_5.g",
"Rb":	"../MCSH_potential_pseudo2/Rb_pseudodensity_5.g",
"Sr":	"../MCSH_potential_pseudo2/Sr_pseudodensity_5.g",
"Y":	"../MCSH_potential_pseudo2/Y_pseudodensity_5.g",
"Zr":	"../MCSH_potential_pseudo2/Zr_pseudodensity_4.g",
"Nb":	"../MCSH_potential_pseudo2/Nb_pseudodensity_4.g",
"Mo":	"../MCSH_potential_pseudo2/Mo_pseudodensity_4.g",
"Tc":	"../MCSH_potential_pseudo2/Tc_pseudodensity_4.g",
"Ru":	"../MCSH_potential_pseudo2/Ru_pseudodensity_4.g",
"Rh":	"../MCSH_potential_pseudo2/Rh_pseudodensity_4.g",
"Pd":	"../MCSH_potential_pseudo2/Pd_pseudodensity_4.g",
"Ag":	"../MCSH_potential_pseudo2/Ag_pseudodensity_4.g",
"Cd":	"../MCSH_potential_pseudo2/Cd_pseudodensity_4.g",
"In":	"../MCSH_potential_pseudo2/In_pseudodensity_4.g",
"Sn":	"../MCSH_potential_pseudo2/Sn_pseudodensity_4.g",
"Sb":	"../MCSH_potential_pseudo2/Sb_pseudodensity_4.g",
"Te":	"../MCSH_potential_pseudo2/Te_pseudodensity_4.g",
"I":	"../MCSH_potential_pseudo2/I_pseudodensity_4.g",
"Cs":	"../MCSH_potential_pseudo2/Cs_pseudodensity_5.g",
"Ba":	"../MCSH_potential_pseudo2/Ba_pseudodensity_5.g",
"Hf":	"../MCSH_potential_pseudo2/Hf_pseudodensity_5.g",
"Ta":	"../MCSH_potential_pseudo2/Ta_pseudodensity_5.g",
"W":	"../MCSH_potential_pseudo2/W_pseudodensity_7.g",
"Re":	"../MCSH_potential_pseudo2/Re_pseudodensity_6.g",
"Os":	"../MCSH_potential_pseudo2/Os_pseudodensity_6.g",
"Ir":	"../MCSH_potential_pseudo2/Ir_pseudodensity_6.g",
"Pt":	"../MCSH_potential_pseudo2/Pt_pseudodensity_6.g",
"Au":	"../MCSH_potential_pseudo2/Au_pseudodensity_6.g",
"Hg":	"../MCSH_potential_pseudo2/Hg_pseudodensity_6.g",
"Tl":	"../MCSH_potential_pseudo2/Tl_pseudodensity_6.g",
"Pb":	"../MCSH_potential_pseudo2/Pb_pseudodensity_6.g",
"Bi":	"../MCSH_potential_pseudo2/Bi_pseudodensity_6.g",}
sigmas_dict = {
    37: [0.02,0.05,0.08,0.12,0.16,0.2,0.24,0.28,0.32,0.36,0.4,0.45,0.5,0.56,0.62,0.69,0.76,0.84,0.92,1.01,1.1,1.2,1.3,1.4,1.52,1.66,1.82,2.0,2.2,2.42,2.66,2.92,3.2,3.5,3.9,4.4,5.0],
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



# elements = ["Cu", "C", "O"]
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

trainer = AtomsTrainer()
trainer.load_pretrained(checkpoint_name, gpu2cpu=True)
trainer.config["dataset"]["save_fps"] = False

result_dirname = "./test_result_val_ood_both/sigma{}_MCSH{}_nodes{}_layers{}_cutoff{}_numtraining{}_results".format(sigmas_index, MCSHs_index, num_nodes,num_layers,cutoff_distance,num_training)

#train_mae = predict_data(trainer, train_images, image_type = "train")
test_mae = predict_data(trainer, test_images, set_index, result_dirname = result_dirname, image_type = "test")


