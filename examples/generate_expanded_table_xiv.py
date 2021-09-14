""" Generate Table XIV from THC costing paper including cost estimates

Note: For Li Hamiltonian, CCSD(T) error estimates are sensitive to converence criteria, etc. So the
      error estimates won't match the paper *exactly* but will be pretty close and trends will hold.

Expected output:

 Double low rank factorization data for 'li_femoco'.
    [*] using CAS((74a, 39b), 76o)
        [+]                      E(SCF):     -1118.65175410
        [+] Active space CCSD(T) E(cor):        -0.25620025
        [+] Active space CCSD(T) E(tot):     -1118.90795435
========================================================================================================================
 threshold        L       eigenvectors    lambda      CCSD(T) error (mEh)       logical qubits       Toffoli count
------------------------------------------------------------------------------------------------------------------------
  0.050000       184          3765        1108.9             -90.76                  3358               3.6e+10
  0.025000       205          5992        1136.0             -20.38                  3363               4.1e+10
  0.012500       247          8450        1152.1             -12.90                  3361               4.7e+10
  0.010000       261          9302        1155.5             -6.96                   3365               4.9e+10
  0.007500       278         10493        1159.1             -5.95                   3366               5.2e+10
  0.005000       312         12508        1163.4             -2.42                   6405               5.6e+10
  0.002500       344         16355        1168.6              1.47                   6406               6.0e+10
  0.001250       394         20115        1171.2              0.05                   6404               6.4e+10
  0.001000       413         21407        1171.7              0.40                   6407               6.6e+10
  0.000750       434         23145        1172.2              0.39                   6405               6.8e+10
  0.000500       470         25751        1172.8              0.40                   6406               7.1e+10
  0.000125       589         35006        1173.7             -0.04                   6410               8.1e+10
  0.000100       614         36557        1173.7             -0.02                   6406               8.3e+10
  0.000050       679         41563        1173.9             -0.01                   6403               8.8e+10
========================================================================================================================
""" 
import sys
from chemftr import df
from chemftr.molecule import load_casfile_to_pyscf

DE = 0.001  # max allowable phase error
CHI = 10    # number of bits for representation of coefficients
BETA = 20   # Value from paper 
THRESH_RANGE = [0.05, 0.025, 0.0125, 0.01, 0.0075, 0.005, 0.0025, 0.00125, 0.001, 0.00075, 0.0005, 
                0.000125, 0.0001, 0.00005] # various DF thresholds

# eri_li.h5 can be found at https://doi.org/10.5281/zenodo.4248322
LI_INTS = '../src/chemftr/integrals/eri_li.h5'  # path to integrals
li_mol, li_mf = load_casfile_to_pyscf(LI_INTS, num_alpha = 74, num_beta = 39)

# This writes out to file "double_factorization_li_femoco.txt"
df.double_factorization(li_mf,name='li_femoco',thresh_range=THRESH_RANGE,dE=DE,chi=CHI,beta=BETA)
