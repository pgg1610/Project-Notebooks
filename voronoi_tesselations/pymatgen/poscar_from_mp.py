__author__ = "Pushkar"
__email__ = "pghaneka@purdue.edu"
__status__ = "Production"
__version__ = "0.1"
__date__ = "Feb, 2019"

from ase.io import read,write
from ase.visualize import view 

import glob
import os
from sys import argv
import subprocess

from pymatgen import MPRester
from pymatgen.io import ase as pm_ase
from pymatgen.io import vasp as pm_vasp 

name = str(argv[1])

MAPI_KEY = 'X7wBIWnT6TdEkHMjvNX'
mpr = MPRester(MAPI_KEY)

base = os.getcwd() 
bridge = pm_ase.AseAtomsAdaptor()

props=["pretty_formula", "material_id", "final_energy", "energy_per_atom","nsites","volume","potcar_symbols"]
#q = mpr.query(criteria={"elements": [name]}, properties=props)
q = mpr.query(criteria={"pretty_formula": name}, properties=props)
for e,n in enumerate(q):
    print(q[e])
    print('***')

'''
for x,y in enumerate(q):
    mp_structure = mpr.get_structure_by_material_id(y['material_id'])
    ase_atoms = bridge.get_atoms(mp_structure)
    ase_atoms.write('POSCAR_'+y['pretty_formula']+'_'+y['material_id'],format='vasp')
'''

q_pool = []
for x,y in enumerate(q):
    q_pool.append([q[x]['energy_per_atom'],q[x]['volume'],q[x]['material_id'],q[x]['pretty_formula']])

q_pool = sorted(q_pool, key = lambda x:x[0], reverse=True)

q_min = [] 
for m,n in enumerate(q_pool):
    min_ = q_pool[-1][0]
    if q_pool[m][0] - min_ <= 0.05:
        q_min.append(q_pool[m])

q_min = sorted(q_min, key = lambda x:x[1], reverse=True)
print('Min energy struct')
print(q_min[-1])
mp_structure = mpr.get_structure_by_material_id(q_min[-1][2],conventional_unit_cell=True)
ase_atoms = bridge.get_atoms(mp_structure)
ase_atoms.write('POSCAR_'+str(q_min[-1][3])+"_"+str(q_min[-1][2]),format='vasp')


'''    
for no, entry in enumerate(pm_atom_list):
    ase_atoms = bridge.get_atoms(entry)
    mk_dir = str(name)+'-'+str(pm_mat_id[no])
    os.makedirs(mk_dir)
    os.chdir(mk_dir)
    ase_atoms = bridge.get_atoms(entry)
    ase_atoms.write('POSCAR',format='vasp')
    vol_k = int(15**3/entry.volume)
    k=pm_vasp.Kpoints.automatic_density_by_vol(structure=entry,kppvol=vol_k)
    k.write_file('KPOINTS')
    subprocess.call("bash ~/bin/chPBE.sh",shell=True)
    subprocess.call("python ~/bin/incar_make.py -m m -xc pbe",shell=True)
    subprocess.call("bash ~/bin/qs_make.sh " + str(mk_dir) + " 1 s 4 vasp", shell=True)
    subprocess.call("qsub vasp.run", shell=True)
    os.chdir(base)
'''
