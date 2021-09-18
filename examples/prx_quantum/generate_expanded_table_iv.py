""" Generate Table IV from THC costing paper including cost estimates

Expected output:

""" 
import sys
from importlib.resources import files
from chemftr import thc 
from chemftr.molecule import load_casfile_to_pyscf
from chemftr.utils import RunSilent

DE = 0.001  # max allowable phase error
CHI = 10    # number of bits for representation of coefficients
BETA = 16

# eri_reiher.h5 can be found at https://doi.org/10.5281/zenodo.4248322
REIHER_INTS = files('chemftr.integrals').joinpath('eri_reiher.h5')  # pre-packaged integrals
reiher_mol, reiher_mf = load_casfile_to_pyscf(REIHER_INTS, num_alpha = 27, num_beta = 27)

# This writes out to file "thc_factorization_reiher_femoco.txt"
thc.generate_costing_table(reiher_mf, name='reiher_femoco', nthc_range=range(250,801,50), dE=DE, chi=CHI, beta=BETA, save_thc=False)
