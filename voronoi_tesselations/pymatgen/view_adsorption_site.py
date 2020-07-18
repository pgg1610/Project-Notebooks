#! /usr/bin/env python
#Use python 3 and pymatgen module to execute this file
#From: https://matgenb.materialsvirtuallab.org/2018/07/24/Adsorption-on-solid-surfaces.html
from pymatgen import Structure, Molecule, MPRester, Lattice
from pymatgen.io import ase as pm_ase
from sys import argv
from ase.io import read, write
from ase.build import add_adsorbate
from ase.visualize import view
from matplotlib import pyplot as plt
import pymatgen.symmetry.analyzer as anal
import pymatgen.analysis.adsorption as py_ads

bridge = pm_ase.AseAtomsAdaptor()
x = read(str(argv[1]))
mp_x = bridge.get_structure(x)
asf = py_ads.AdsorbateSiteFinder(mp_x)
ads_sites = asf.find_adsorption_sites(symm_reduce=1.1E-2,near_reduce=1E-2)
print(len(ads_sites['all']))
print(ads_sites['all'])

adsorbate = Molecule("H", [[0, 0, 0]])
for i,j in enumerate(ads_sites['all']):
    print(j)
    asf = py_ads.AdsorbateSiteFinder(mp_x)
    _ads_mp=asf.add_adsorbate(adsorbate,j)
    mp_x=_ads_mp  

_ads_ase=bridge.get_atoms(_ads_mp)
view(_ads_ase)


