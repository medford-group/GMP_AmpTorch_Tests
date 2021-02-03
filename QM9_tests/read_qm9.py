import json

ref_energy_dict = {}
filename = "./reference_energy.txt"
with open(filename) as fp:
    Lines = fp.readlines()
    for line in Lines[4:9]:
        temp = line.strip().split()
        atom = temp[0]
        ref_energy_dict[atom] = {}
        ref_energy_dict[atom]["u0"] =   float(temp[2]) 
        ref_energy_dict[atom]["u298"] = float(temp[3]) 
        ref_energy_dict[atom]["h298"] = float(temp[4]) 
        ref_energy_dict[atom]["g298"] = float(temp[5])
print(ref_energy_dict)


data = {}

for i in range(1,133886):

    system_data = {}
    coordinates = []
    atoms = []
    system_name = "gdb_{}".format(i)
    filename = f"raw_data/dsgdb9nsd_{i:06}.xyz"
    with open(filename) as fp:
        Lines = fp.readlines()
        num_atom = int(Lines[0].strip())
        temp = Lines[1].strip().split()
        u0 =   float(temp[12])
        u298 = float(temp[13])
        h298 = float(temp[14])
        g298 = float(temp[15])
        u0_atom =   float(temp[12])
        u298_atom = float(temp[13])
        h298_atom = float(temp[14])
        g298_atom = float(temp[15])
        #print(u0_atom)
        #print(num_atom)
        for line in Lines[2:-3]:
            temp_atom = line.replace("*^","e").strip().split()
            atom = temp_atom[0]
            atoms.append(temp_atom[0])
            u0_atom -=   ref_energy_dict[atom]["u0"]
            #print("**{}\t{}\t{}".format(atom, ref_energy_dict[atom]["u0"], u0_atom))
            u298_atom -= ref_energy_dict[atom]["u298"]
            h298_atom -= ref_energy_dict[atom]["h298"]
            g298_atom -= ref_energy_dict[atom]["g298"]
            coordinates.append([float(temp_atom[1]), float(temp_atom[2]), float(temp_atom[3])])
        assert len(atoms) == num_atom
        #print(u0_atom)
        system_data["coordinates"] = coordinates
        system_data["atoms"] = atoms
        system_data["u0"]   = u0
        system_data["u298"] = u298
        system_data["h298"] = h298
        system_data["g298"] = g298
        system_data["u0_atom"]   = u0_atom * 627.5
        system_data["u298_atom"] = u298_atom * 627.5
        system_data["h298_atom"] = h298_atom * 627.5
        system_data["g298_atom"] = g298_atom * 627.5
        data[system_name] = system_data


with open('qm9.json', 'w') as fp:
    json.dump(data, fp, indent = 4)