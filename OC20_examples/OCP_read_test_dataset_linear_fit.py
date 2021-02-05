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


def apply_linear_fit(dataset, test_list):

    correction_dict = load_linear_fit_result("./linear_model_result.dat")
    atom_list = correction_dict.keys()


    test_atom_count_list = []
    print("start counting atoms")


    for i,system in enumerate(test_list):
        temp_atoms = system.get_chemical_symbols()
        temp_atom_count_dict = {}
        for atom in temp_atoms:
            temp_atom_count_dict[atom] = temp_atom_count_dict.get(atom, 0) + 1


        test_atom_count_list.append(temp_atom_count_dict)

    result2 = ""

    corrected_energy_list = []
    atoms_count_list = []
    for i, system in enumerate(test_list):
        energy = system.get_calculator().get_potential_energy()
        num_atoms = 0
        for atom in test_atom_count_list[i].keys():
            energy -= test_atom_count_list[i][atom] * correction_dict[atom]
            num_atoms += test_atom_count_list[i][atom]
        corrected_energy_list.append(energy)
        calc = SinglePointCalculator(system,energy = energy, forces = system.get_calculator().get_forces())
        system.set_calculator(calc)
        atoms_count_list.append(num_atoms)

    corrected_energy_list = np.array(corrected_energy_list)
    atoms_count_list = np.array(atoms_count_list)
    corrected_energy_per_atom_list = np.divide(corrected_energy_list, atoms_count_list)

    result2 += "test me: {}\n".format(np.mean(corrected_energy_list))
    result2 += "test mae: {}\n".format(np.mean(np.abs(corrected_energy_list)))
    result2 += "test mse: {}\n".format(np.mean(np.square(corrected_energy_list)))

    result2 += "test mepa: {}\n".format(np.mean(corrected_energy_per_atom_list))
    result2 += "test maepa: {}\n".format(np.mean(np.abs(corrected_energy_per_atom_list)))
    result2 += "test msepa: {}\n".format(np.mean(np.square(corrected_energy_per_atom_list)))
    
    result_filename = "linear_result_{}.dat".format(dataset)
    log(result_filename,result2)


    return test_list


def load_training_data(dataset, test_filename):


    test_data_list = []

    for i in range(200):
        print(i)
        filename = "../{}/{}/{}.extxyz".format(dataset, dataset, i)
        test_data_list += read(filename, index='0:4999')

    energy_list_pre = []
    for image in test_data_list:
        image.center()
        #atom_list += image.get_chemical_symbols()
        energy_list_pre.append(image.get_calculator().get_potential_energy())

    print("average energy before: ")
    print(np.mean(energy_list_pre))

    test_images = apply_linear_fit(dataset, test_data_list)

    energy_list_after = [image.get_calculator().get_potential_energy() for image in test_images]
    print("average energy after: ")
    print(np.mean(energy_list_after))
    pickle.dump( test_images, open( test_filename, "wb" ) )

    return test_images

torch.set_num_threads(1)
trail_num = sys.argv[1]
dataset = sys.argv[2]
test_filename = "OCP_test_{}.p".format(dataset)
folder_name = "trial_{}".format(trail_num)

os.chdir(folder_name)
load_training_data(dataset, test_filename)
