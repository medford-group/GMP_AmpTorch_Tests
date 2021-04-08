import pickle
import os
import torch
import lmdb
import numpy as np
import ase.io
from tqdm import tqdm
from amptorch.preprocessing import AtomsToData, FeatureScaler, TargetScaler
from amptorch.descriptor.Gaussian import Gaussian
from amptorch.descriptor.GMP import GMP
from ase import Atoms
from ase.calculators.emt import EMT

from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
import pickle
import sys
import os


def construct_lmdb(images, normalizers, lmdb_path, sigmas_index, MCSHs_index):
    """
    path: Path to trajectory/ASE-compatible file
    lmdb_path: Path to store LMDB dataset.
    """


    db = lmdb.open(
        lmdb_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

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
                "cutoff": 15.0
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


    forcetraining = False
    descriptor = GMP(MCSHs=MCSHs, elements=elements)
    descriptor_setup = ("gmp", MCSHs, {}, elements)

    a2d = AtomsToData(
        descriptor=descriptor,
        r_energy=True,
        r_forces=False,
        save_fps=False,
        fprimes=forcetraining,
    )

    data_list = []
    idx = 0
    #images = ase.io.read(path, ":")
    for image in tqdm(
                images,
                desc="converting images",
                total=len(images),
                unit=" images",
            ):
    #for image in images:
        do = a2d.convert(image, idx=idx)
        data_list.append(do)
        idx += 1

    feature_scaler = normalizers["feature"]
    target_scaler = normalizers["target"]

    feature_scaler.norm(data_list)
    target_scaler.norm(data_list)

    idx = 0
    for do in tqdm(data_list, desc="Writing images to LMDB"):
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(do, protocol=-1))
        txn.commit()
        idx += 1

    txn = db.begin(write=True)
    txn.put("feature_scaler".encode("ascii"), pickle.dumps(feature_scaler, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put("target_scaler".encode("ascii"), pickle.dumps(target_scaler, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put("elements".encode("ascii"), pickle.dumps(elements, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put(
        "descriptor_setup".encode("ascii"), pickle.dumps(descriptor_setup, protocol=-1)
    )
    txn.commit()

    db.sync()
    db.close()


if __name__ == "__main__":
    sigmas_index = int(sys.argv[1])
    MCSHs_index = int(sys.argv[2])
    dataset_index = int(sys.argv[3])
    os.chdir("./trial_data_full_20M")
    training_data = pickle.load( open( "./images/images_s2ef_train_20M_{}.p".format(dataset_index), "rb" ) )
    lmdb_data_dirname = "lmdbs_sigma{}_MCSH{}/".format(sigmas_index,MCSHs_index)
    lmdb_data_filename = lmdb_data_dirname+ "{}.lmdb".format(dataset_index)
    normalizers = torch.load("normalizers_{}_{}.pt".format(sigmas_index, MCSHs_index))
    construct_lmdb(training_data, normalizers,lmdb_path=lmdb_data_filename, sigmas_index = sigmas_index, MCSHs_index = MCSHs_index)