""" Generate Table XI from THC costing paper including cost estimates

Expected output:

 Single low rank factorization data for 'reiher_femoco'.
    [*] using CAS((27a, 27b), 54o)
        [+]                      E(SCF):    -13481.66850940
        [+] Active space CCSD(T) E(cor):        -0.50689291
        [+] Active space CCSD(T) E(tot):    -13482.17540230
=========================================================================================
     L          lambda      CCSD(T) error (mEh)       logical qubits       Toffoli count    
-----------------------------------------------------------------------------------------
     50         3367.6              4.79                   1779               4.0e+10       
     75         3657.9              0.21                   1782               5.2e+10       
    100         3854.3              1.55                   1782               6.3e+10       
    125         3997.4              3.08                   1782               7.4e+10       
    150         4112.7              2.07                   3320               8.2e+10       
    175         4199.2              1.63                   3320               8.8e+10       
    200         4258.0              0.10                   3320               9.5e+10       
    225         4300.7              0.38                   3320               1.0e+11       
    250         4331.9              0.16                   3320               1.1e+11       
    275         4354.9              0.29                   3323               1.1e+11       
    300         4372.0              0.18                   3323               1.2e+11       
    325         4385.3              0.13                   3323               1.2e+11       
    350         4395.6              0.06                   3323               1.3e+11       
    375         4403.4              0.03                   3323               1.3e+11       
    400         4409.3              0.02                   3323               1.4e+11       
=========================================================================================
""" 
import sys
from importlib.resources import files
from chemftr import sf
from chemftr.molecule import load_casfile_to_pyscf
from chemftr.utils import RunSilent

DE = 0.001  # max allowable phase error
CHI = 10    # number of bits for representation of coefficients

# eri_reiher.h5 can be found at https://doi.org/10.5281/zenodo.4248322
REIHER_INTS = files('chemftr.integrals').joinpath('eri_reiher.h5')  # pre-packaged integrals
reiher_mol, reiher_mf = load_casfile_to_pyscf(REIHER_INTS, num_alpha = 27, num_beta = 27)

with RunSilent():  # context manager to silence printing to stdout
    # This writes out to file "single_factorization_reiher_femoco.txt"
    sf.generate_costing_table(reiher_mf, name='reiher_femoco',chi=CHI,dE=DE)
