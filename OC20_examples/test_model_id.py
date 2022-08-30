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

    log(result_dirname + '/test_mae_id.dat', '{}\t{}\n'.format(set_index,mae_result))
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




test_filename = "s2ef_val_id_images/{}.p".format(set_index)
folder_name = "trial_{}".format(trail_num)

os.chdir(folder_name)


test_images = pickle.load( open( test_filename, "rb" ) )


trainer = AtomsTrainer()
trainer.load_pretrained(checkpoint_name, gpu2cpu=True)
trainer.config["dataset"]["save_fps"] = False

result_dirname = "./test_result_val_id/sigma{}_MCSH{}_nodes{}_layers{}_cutoff{}_numtraining{}_results".format(sigmas_index, MCSHs_index, num_nodes,num_layers,cutoff_distance,num_training)

test_mae = predict_data(trainer, test_images, set_index, result_dirname = result_dirname, image_type = "test")


