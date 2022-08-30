# importing the module 
from ase.io import read
import json 
import ase
import numpy as np
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
#from amptorch.trainer import AtomsTrainer
import pickle
import random
import math
import sys
#import torch
import os
#from sklearn.linear_model import LinearRegression
​
​
def log(log_filename, message):
    f = open(log_filename, "a")
    f.write(message)
    f.close()
    return
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
#     print(correction_dict)        
    return correction_dict
​
​
​
def read_correction_file(filename):
    result = []
    with open(filename) as fp:
        Lines = fp.readlines()
        for line in Lines:
            temp = line.strip().split(',')
            result.append(float(temp[-1]))
    return result
​
​
def apply_system_correction(training_list, correction_dict, correction_list, image_index):
    training_atom_count_list = []
    print("start counting atoms")
​
​
    for i,system in enumerate(training_list):
        temp_atoms = system.get_chemical_symbols()
        temp_atom_count_dict = {}
        for atom in temp_atoms:
            temp_atom_count_dict[atom] = temp_atom_count_dict.get(atom, 0) + 1
​
        training_atom_count_list.append(temp_atom_count_dict)
​
​
    for i, system in enumerate(training_list):
        system.center()
​
        energy = system.get_calculator().get_potential_energy()
        energy = energy - correction_list[i]
        for atom in training_atom_count_list[i].keys():
            energy -= training_atom_count_list[i][atom] * correction_dict[atom]
​
        system.set_pbc([True, True, True])
        calc = SinglePointCalculator(system,energy = energy, forces = system.get_calculator().get_forces())
        system.set_calculator(calc)
    return training_list
​
​
def load_training_data(dataset,correction_dict,index):
​
    #atom_list = []
    #training_data_list = []
    result = []
    for i in range(index * 10, (index+1) * 10):
        print("================= start image {} =====================".format(i))
        filename = "../{}/{}/{}.extxyz".format(dataset,dataset,i)
        filename2 = "../{}/{}/{}.txt".format(dataset,dataset,i)
        training_data_list = read(filename, index=':')
        correction_list = read_correction_file(filename2)
​
        print(len(training_data_list))
​
        training_images = apply_system_correction(training_data_list, correction_dict, correction_list, i)
        result += training_images
        energy_list_after = [image.get_calculator().get_potential_energy() for image in training_images]
        print("average energy after: ")
        print(np.mean(energy_list_after))
​
    training_filename = "images/images_{}_{}.p".format(dataset,index)
    pickle.dump( result , open( training_filename, "wb" ) )
​
    return 
​
trail_num = "data_full_20M_linear_correct"
dataset = "s2ef_train_20M"
folder_name = "trial_{}".format(trail_num)
os.chdir(folder_name)
correction_dict = load_linear_fit_result("linear_model_result.dat")
for index in range(20):
    print("**** start index {} ****".format(index))
    load_training_data(dataset,correction_dict,index)