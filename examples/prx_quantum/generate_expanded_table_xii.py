""" Generate Table XII from THC costing paper including cost estimates

Note: For Li Hamiltonian, CCSD(T) error estimates are sensitive to converence criteria, etc. So the
      error estimates won't match the paper *exactly* but will be pretty close and trends will hold.

Expected output:

 Single low rank factorization data for 'li_femoco'.
    [*] using CAS((74a, 39b), 76o)
        [+]                      E(SCF):     -1118.65175412
        [+] Active space CCSD(T) E(cor):        -0.25620522
        [+] Active space CCSD(T) E(tot):     -1118.90795934
=========================================================================================
     L          lambda      CCSD(T) error (mEh)       logical qubits       Toffoli count
-----------------------------------------------------------------------------------------
     50         2233.7             -91.51                  1954               3.8e+10
     75         2484.5             -56.07                  3620               5.2e+10
    100         2664.5             -23.11                  3620               6.2e+10
    125         2743.5             -9.15                   3622               7.0e+10
    150         2786.9             -11.38                  3625               7.8e+10
    175         2835.5             -8.10                   3625               8.5e+10
    200         2906.9              0.43                   3625               9.4e+10
    225         2986.9              1.33                   3625               1.0e+11
    250         3035.9              0.80                   3625               1.1e+11
    275         3071.8              0.43                   3628               1.2e+11
    300         3099.2             -0.11                   6955               1.2e+11
    325         3119.3             -0.20                   6955               1.3e+11
    350         3134.2             -0.12                   6955               1.3e+11
    375         3146.0             -0.09                   6955               1.4e+11
    400         3154.8             -0.10                   6955               1.4e+11
=========================================================================================
""" 
import sys
from importlib.resources import files
from chemftr import sf
from chemftr.molecule import load_casfile_to_pyscf, stability, pyscf_to_cas
from chemftr.utils import RunSilent

DE = 0.001  # max allowable phase error
CHI = 10    # number of bits for representation of coefficients

# eri_li.h5 can be found at https://doi.org/10.5281/zenodo.4248322
LI_INTS = files('chemftr.integrals').joinpath('eri_li.h5')  # pre-packaged integrals
li_mol, li_mf = load_casfile_to_pyscf(LI_INTS, num_alpha = 74, num_beta = 39)

with RunSilent():
    # This writes out to file "single_factorization_li_femoco.txt"
    sf.generate_costing_table(li_mf, name='li_femoco',chi=CHI,dE=DE)
