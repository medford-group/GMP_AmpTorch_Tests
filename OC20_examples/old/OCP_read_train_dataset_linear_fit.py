# importing the module 
from ase.io import read
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
from sklearn.linear_model import LinearRegression


def log(log_filename, message):
    f = open(log_filename, "a")
    f.write(message)
    f.close()
    return

def linear_fit(training_list, atom_list):
    atom_dict = {atom:i for i, atom in enumerate(atom_list)}
    atom_dict_rev = {i:atom for i, atom in enumerate(atom_list)}

    # atom_count_list = []
    

    training_atom_count_list = []
    print("start counting atoms")


    for i,system in enumerate(training_list):
        temp_atoms = system.get_chemical_symbols()
        temp_atom_count_dict = {}
        for atom in temp_atoms:
            temp_atom_count_dict[atom] = temp_atom_count_dict.get(atom, 0) + 1


        training_atom_count_list.append(temp_atom_count_dict)



    print("start preparing linear system")
    num_atom_mat = np.zeros((len(training_list), len(atom_list)))
    energy_vec = np.zeros((len(training_list),)) 
    for i, system in enumerate(training_list):
#         energy_vec[i] = system.get_potential_energy()
        energy_vec[i] = system.get_calculator().get_potential_energy()
        atom_counts = training_atom_count_list[i]
        for atom in atom_counts.keys():
            num_atom_mat[i, atom_dict[atom]] = atom_counts[atom]


    print(num_atom_mat)
    print(energy_vec)
    reg = LinearRegression().fit(num_atom_mat, energy_vec)
    linear_regression_result = reg.coef_
    correction_dict = {atom_dict_rev[i]: correction for i, correction in enumerate(linear_regression_result)}
    print(correction_dict)

    linear_model_result_filename = "linear_model_result.dat"
    linear_model_info_filename = "linear_model_info.dat"

    result1 = ""
    result2 = ""

    for atom in correction_dict.keys():
        result1 += "{}\t{}\n".format(atom, correction_dict[atom])
    log(linear_model_result_filename,result1)
    
    
    corrected_energy_list = []
    atoms_count_list = []
    for i, system in enumerate(training_list):
        energy = system.get_calculator().get_potential_energy()
        num_atoms = 0
        for atom in training_atom_count_list[i].keys():
            energy -= training_atom_count_list[i][atom] * correction_dict[atom]
            num_atoms += training_atom_count_list[i][atom]
        corrected_energy_list.append(energy)
        calc = SinglePointCalculator(system,energy = energy, forces = system.get_calculator().get_forces())
        system.set_calculator(calc)
        atoms_count_list.append(num_atoms)

    corrected_energy_list = np.array(corrected_energy_list)
    atoms_count_list = np.array(atoms_count_list)
    corrected_energy_per_atom_list = np.divide(corrected_energy_list, atoms_count_list)

    result2 += "training me: {}\n".format(np.mean(corrected_energy_list))
    result2 += "training mae: {}\n".format(np.mean(np.abs(corrected_energy_list)))
    result2 += "training mse: {}\n".format(np.mean(np.square(corrected_energy_list)))

    result2 += "training mepa: {}\n".format(np.mean(corrected_energy_per_atom_list))
    result2 += "training maepa: {}\n".format(np.mean(np.abs(corrected_energy_per_atom_list)))
    result2 += "training msepa: {}\n".format(np.mean(np.square(corrected_energy_per_atom_list)))



    log(linear_model_info_filename,result2)
    

    print("end linear fit")
    return training_list


def load_training_data(dataset, training_filename):



    atom_list = []
    training_data_list = []
    for i in range(40):
        print(i)
        filename = "../{}/{}/{}.extxyz".format(dataset,dataset,i)
        training_data_list += read(filename, index='0:4999')


    energy_list_pre = []
    for image in training_data_list:
        image.center()
        atom_list += image.get_chemical_symbols()
        energy_list_pre.append(image.get_calculator().get_potential_energy())

    atom_list = list(set(atom_list))

    print("average energy before: ")
    print(np.mean(energy_list_pre))

    training_images = linear_fit(training_data_list, atom_list)
    energy_list_after = [image.get_calculator().get_potential_energy() for image in training_images]
    print("average energy after: ")
    print(np.mean(energy_list_after))

    pickle.dump( training_images, open( training_filename, "wb" ) )

    return 

torch.set_num_threads(1)
trail_num = sys.argv[1]
dataset = "s2ef_train_200K"
train_filename = "OCP_train_{}.p".format(dataset)
folder_name = "trial_{}".format(trail_num)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
os.chdir(folder_name)
load_training_data(dataset,train_filename)
