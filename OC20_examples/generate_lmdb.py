import pickle
import os
import torch
import lmdb
import numpy as np
import ase.io
from tqdm import tqdm
from amptorch.preprocessing import AtomsToData, FeatureScaler, TargetScaler
from amptorch.descriptor.Gaussian import Gaussian
from amptorch.descriptor.GMPOrderNorm import GMPOrderNorm
from ase import Atoms
from ase.calculators.emt import EMT
​
from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
import pickle
import sys
import os
​
​
def construct_lmdb(images, normalizers, lmdb_path, nsigmas, MCSHs_index):
    """
    path: Path to trajectory/ASE-compatible file
    lmdb_path: Path to store LMDB dataset.
    """
​
    db = lmdb.open(
        lmdb_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
​
    elements = ['Pt','Al','V','Pd','Fe','Sn','Ge','Bi','Ir','Re','Cd','Cr','Ag','Hf','Ru','Ti','Cs','Os','N',
     'As','O','S','Mo','Ta','Zn','Y','Mn','Na','Rh','Hg','C','Co','Nb','Sc','Sr','H','Au','Ga','Tl','K','Se',
     'B','Pb','Ca','Cl','Cu','Zr','Rb','P','W','Tc','Te','Ni','Sb','Si','In']
​
    potential_files = {element: "../../psp_pseudo_v3/{}_pseudodensity.g".format(element) for element in elements}
​
    sigmas = np.linspace(0.02,2.0,nsigmas,endpoint=True)
​
    MCSHs_dict = {
        0: { "orders": [0], "sigmas": sigmas,},
        1: { "orders": [0,1], "sigmas": sigmas,},
        2: { "orders": [0,1,2], "sigmas": sigmas,},
        3: { "orders": [0,1,2,3], "sigmas": sigmas,},
        4: { "orders": [0,1,2,3,4], "sigmas": sigmas,},
        5: { "orders": [0,1,2,3,4,5], "sigmas": sigmas,},
        6: { "orders": [0,1,2,3,4,5,6], "sigmas": sigmas,},
        7: { "orders": [0,1,2,3,4,5,6,7], "sigmas": sigmas,},
        8: { "orders": [0,1,2,3,4,5,6,7,8], "sigmas": sigmas,},
        9: { "orders": [0,1,2,3,4,5,6,7,8,9], "sigmas": sigmas,},
    }
​
    MCSHs = MCSHs_dict[MCSHs_index]
​
    MCSHs = {   "MCSHs": MCSHs,
                "atom_gaussians": potential_files,
                "cutoff": 10.0, 
                "square":False,
                "solid_harmonics": True,
    }
​
    forcetraining = False
    descriptor = GMPOrderNorm(MCSHs=MCSHs, elements=elements)
    descriptor_setup = ("gmpordernorm", MCSHs, {}, elements)
​
    a2d = AtomsToData(
        descriptor=descriptor,
        r_energy=True,
        r_forces=False,
        save_fps=True,
        fprimes=forcetraining,
    )
​
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
​
    feature_scaler = normalizers["feature"]
    target_scaler = normalizers["target"]
​
    feature_scaler.norm(data_list)
    target_scaler.norm(data_list)
​
    idx = 0
    for do in tqdm(data_list, desc="Writing images to LMDB"):
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(do, protocol=-1))
        txn.commit()
        idx += 1
​
    txn = db.begin(write=True)
    txn.put("feature_scaler".encode("ascii"), pickle.dumps(feature_scaler, protocol=-1))
    txn.commit()
​
    txn = db.begin(write=True)
    txn.put("target_scaler".encode("ascii"), pickle.dumps(target_scaler, protocol=-1))
    txn.commit()
​
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()
​
    txn = db.begin(write=True)
    txn.put("elements".encode("ascii"), pickle.dumps(elements, protocol=-1))
    txn.commit()
​
    txn = db.begin(write=True)
    txn.put(
        "descriptor_setup".encode("ascii"), pickle.dumps(descriptor_setup, protocol=-1)
    )
    txn.commit()
​
    db.sync()
    db.close()
​
​
if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    nsigmas = int(sys.argv[1])
    MCSHs_index = int(sys.argv[2])
    dataset_index = int(sys.argv[3])
    os.chdir("./trial_data_full_20M")
    #dirname = sys.argv[4]
    #os.chdir(dirname)
    training_data = pickle.load( open( "./images/images_s2ef_train_20M_{}.p".format(dataset_index), "rb" ) )
    lmdb_data_dirname = "lmdbs_sigma{}_MCSH{}_double/".format(nsigmas ,MCSHs_index)
    lmdb_data_filename = lmdb_data_dirname+ "{}.lmdb".format(dataset_index)
    normalizers = torch.load("normalizers_{}_{}_double.pt".format(nsigmas , MCSHs_index))
    construct_lmdb(training_data, normalizers,lmdb_path=lmdb_data_filename, nsigmas = nsigmas, MCSHs_index = MCSHs_index)