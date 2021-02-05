import numpy as np
import json 
import ase
from ase import Atoms, Atom
from ase.calculators.singlepoint import SinglePointCalculator
import random
import pickle


def load_images(filename = "aspirin.xyz", num_atoms = 21, element_list = ["C","H","O"], energy_offset = 0.0):
    images = []


    atom_list = []
    read_energy = False
    energy = 0
    first = True
    with open(filename) as fp: 
        Lines = fp.readlines() 
        for line in Lines:
            temp = line.strip().split()
            if len(temp) == 1 and read_energy:
                energy = float(temp[0]) - energy_offset
#                 print(energy)
                read_energy = False
            elif len(temp) == 1 and int(temp[0]) == num_atoms:
                if first == False:
                    system = Atoms(atom_list, cell=[50, 50, 50])
                    system.center()
    #                 print(system.get_atomic_numbers())
                    calc = SinglePointCalculator(system,energy = energy)
                    system.set_calculator(calc)
                    images.append(system)
                else:
                    first = False
                atom_list = []
                read_energy = True
            elif len(temp) == 7 and temp[0] in element_list:
                element_type = temp[0]
                x = float(temp[1])
                y = float(temp[2])
                z = float(temp[3])
                atom_list.append(Atom(element_type, np.array([x,y,z])))
            else:
                print("line read error")
    return images
 

def get_dataset(images, system_name, num_train = 10000, num_test=10000, trial = 1):
    index_list = random.sample(range(0, len(images)-1), num_train + num_test)
    train_index_list = index_list[:num_train]
    test_index_list  = index_list[num_train:]
    train_list = [images[i] for i in train_index_list]
    test_list = [images[i] for i in test_index_list]
    
    train_filename = "data/{}_train_data_{}.p".format(system_name, trial)
    test_filename = "data/{}_test_data_{}.p".format(system_name, trial)
    
    pickle.dump( train_list, open( train_filename, "wb" ) )
    pickle.dump( test_list, open( test_filename, "wb" ) )
    return

images = load_images(filename = "aspirin.xyz", num_atoms = 21, element_list = ["C","H","O"])


get_dataset(images, "aspirin", num_train = 50000, num_test=10000, trial = 1)
get_dataset(images, "aspirin", num_train = 50000, num_test=10000, trial = 2)
get_dataset(images, "aspirin", num_train = 50000, num_test=10000, trial = 3)
get_dataset(images, "aspirin", num_train = 50000, num_test=10000, trial = 4)
get_dataset(images, "aspirin", num_train = 50000, num_test=10000, trial = 5)
get_dataset(images, "aspirin", num_train = 50000, num_test=10000, trial = 6)
get_dataset(images, "aspirin", num_train = 50000, num_test=10000, trial = 7)
get_dataset(images, "aspirin", num_train = 50000, num_test=10000, trial = 8)
get_dataset(images, "aspirin", num_train = 50000, num_test=10000, trial = 9)
get_dataset(images, "aspirin", num_train = 50000, num_test=10000, trial = 10)