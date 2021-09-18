""" Generate Table V from THC costing paper including cost estimates

Expected output:

""" 
import sys
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files
from chemftr import thc 
from chemftr.molecule import load_casfile_to_pyscf
from chemftr.utils import RunSilent

DE = 0.001  # max allowable phase error
CHI = 10    # number of bits for representation of coefficients
BETA = 20

# eri_LI.h5 can be found at https://doi.org/10.5281/zenodo.4248322
LI_INTS = files('chemftr.integrals').joinpath('eri_li.h5')  # pre-packaged integrals
li_mol, li_mf = load_casfile_to_pyscf(LI_INTS, num_alpha = 74, num_beta = 39)

# This writes out to file "thc_factorization_LI_femoco.txt"
thc.generate_costing_table(li_mf, name='li_femoco', nthc_range=range(250,801,50), dE=DE, chi=CHI, beta=BETA, save_thc=False)
