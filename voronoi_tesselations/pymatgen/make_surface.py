#! /usr/bin/env python
#Use python 3 and pymatgen module to execute this file
from sys import argv
from ase.io import read, write 
from ase.build import surface 
from ase.visualize import view

bulk = read(str(argv[1]))
facet=(int(argv[2]),int(argv[3]),int(argv[4]))
slab = surface(bulk, facet,int(argv[5]))
slab.center(vacuum=10, axis=2)
view(slab)
#write(str(argv[1])+'.surface',slab, format='vasp')
