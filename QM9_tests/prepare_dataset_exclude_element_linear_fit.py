# importing the module 
import json 
import ase
import numpy as np
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
# from amptorch.trainer import AtomsTrainer
import pickle
import random
import math
import sys
# import torch
import os
from sklearn.linear_model import LinearRegression


def log(log_filename, message):
    f = open(log_filename, "a")
    f.write(message)
    f.close()
    return

def load_data_to_atoms(system_data):
    atoms_list = []
    for i, atom in enumerate(system_data["atoms"]):
        atoms_list.append(Atom(atom, np.array(system_data["coordinates"][i])))
    system = Atoms(atoms_list, cell=[50, 50, 50])
    system.center()
    calc = SinglePointCalculator(system,energy = system_data["corrected_energy"])
    system.set_calculator(calc)
    return system




def linear_fit(data_list, atom_list, exclude_atom = ["F"]):
    atom_dict = {atom:i for i, atom in enumerate(atom_list)}
    atom_dict_rev = {i:atom for i, atom in enumerate(atom_list)}    

    training_list = []
    test_list = []
    training_atom_count_list = []
    test_atom_count_list = []
    overall_atom_count_list = []
    print("start counting atoms")
    for i,system in enumerate(data_list):
        training = True
        temp_atoms = system["atoms"]
        temp_atom_count_dict = {}
        for atom in temp_atoms:
            if atom in exclude_atom:
                training = False
            
            temp_atom_count_dict[atom] = temp_atom_count_dict.get(atom, 0) + 1

        overall_atom_count_list.append(temp_atom_count_dict)

        if training:
            training_list.append(system)
            training_atom_count_list.append(temp_atom_count_dict)
        else:
            test_list.append(system)
            test_atom_count_list.append(temp_atom_count_dict)
    
    print("training list length: {}".format(len(training_list)))
    print("test list length: {}".format(len(test_list)))


    # print("start preparing linear system")
    # num_atom_mat = np.zeros((len(training_list), len(atom_list)))
    # energy_vec = np.zeros((len(training_list),)) 
    # for i, system in enumerate(training_list):
    #     energy_vec[i] = system["u0"]
    #     atom_counts = training_atom_count_list[i]
    #     for atom in atom_counts.keys():
    #         num_atom_mat[i, atom_dict[atom]] = atom_counts[atom]


    # print(num_atom_mat)
    # print(energy_vec)
    # reg = LinearRegression().fit(num_atom_mat, energy_vec)
    # linear_regression_result = reg.coef_
    # correction_dict = {atom_dict_rev[i]: correction for i, correction in enumerate(linear_regression_result)}
    correction_dict =  {"H": -65.00632675636872,
                        "C":   -144.59934607239086,
                        "N":   -105.87077077504317,
                        "O":   -103.08549133714506,
                        "F":   -94.7789104315264}

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
        energy = system["u0_atom"]
        num_atoms = 0
        for atom in training_atom_count_list[i].keys():
            energy -= training_atom_count_list[i][atom] * correction_dict[atom]
            num_atoms += training_atom_count_list[i][atom]
        corrected_energy_list.append(energy)
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





    corrected_energy_list = []
    atoms_count_list = []
    for i, system in enumerate(test_list):
        energy = system["u0_atom"]
        num_atoms = 0
        for atom in test_atom_count_list[i].keys():
            energy -= test_atom_count_list[i][atom] * correction_dict[atom]
            num_atoms += test_atom_count_list[i][atom]
        corrected_energy_list.append(energy)
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
    

    




    corrected_energy_list = []
    atoms_count_list = []
    for i, system in enumerate(data_list):
        system["corrected_energy"] = system["u0_atom"]
        num_atoms = 0
        for atom in overall_atom_count_list[i].keys():
            system["corrected_energy"] -= overall_atom_count_list[i][atom] * correction_dict[atom]
            num_atoms += overall_atom_count_list[i][atom]
        corrected_energy_list.append(system["corrected_energy"])
        atoms_count_list.append(num_atoms)

    corrected_energy_list = np.array(corrected_energy_list)
    atoms_count_list = np.array(atoms_count_list)
    corrected_energy_per_atom_list = np.divide(corrected_energy_list, atoms_count_list)

    result2 += "overall me: {}\n".format(np.mean(corrected_energy_list))
    result2 += "overall mae: {}\n".format(np.mean(np.abs(corrected_energy_list)))
    result2 += "overall mse: {}\n".format(np.mean(np.square(corrected_energy_list)))

    result2 += "overall mepa: {}\n".format(np.mean(corrected_energy_per_atom_list))
    result2 += "overall maepa: {}\n".format(np.mean(np.abs(corrected_energy_per_atom_list)))
    result2 += "overall msepa: {}\n".format(np.mean(np.square(corrected_energy_per_atom_list)))

    log(linear_model_info_filename,result2)
    

    return training_list, test_list


def load_training_data(training_filename, test_filename, atom_list = ["H","C","N","O","F"]):

    try:
        training_list = pickle.load( open( training_filename, "rb" ) )
        test_list = pickle.load( open( test_filename, "rb" ) )
    except:

        with open('../qm9.json') as json_file: 
            data = json.load(json_file)

        data_list = []
        for i, system in enumerate(data.keys()):
            data_list.append(data[system])


        training_list, test_list = linear_fit(data_list, atom_list)

        training_images = [load_data_to_atoms(system) for system in training_list]
        test_images = [load_data_to_atoms(system) for system in test_list]

        images = []
        for i, system in enumerate(data.keys()):
            image = load_data_to_atoms(data[system])
            images.append(image)

        pickle.dump( training_images, open( training_filename, "wb" ) )
        pickle.dump( test_images, open( test_filename, "wb" ) )

    return training_images, test_images


# torch.set_num_threads(1)
trail_num = "exclude_atom_F_linear_fit"
train_filename = "QM9_train_{}.p".format(trail_num)
test_filename = "QM9_test_{}.p".format(trail_num)
folder_name = "trial_{}".format(trail_num)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
os.chdir(folder_name)
train_images, test_images = load_training_data(train_filename, test_filename)