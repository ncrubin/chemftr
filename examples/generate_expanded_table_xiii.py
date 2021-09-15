""" Generate Table XIII from THC costing paper including cost estimates

Expected output:

 Double low rank factorization data for 'reiher_femoco'.
    [*] using CAS((27a, 27b), 54o)
        [+]                      E(SCF):    -13481.66850940
        [+] Active space CCSD(T) E(cor):        -0.50689292
        [+] Active space CCSD(T) E(tot):    -13482.17540231
========================================================================================================================
 threshold        L       eigenvectors    lambda      CCSD(T) error (mEh)       logical qubits       Toffoli count
------------------------------------------------------------------------------------------------------------------------
  0.050000        95          1274        262.1              -12.39                  1123               4.3e+09
  0.025000       135          2319        272.0              -2.78                   1991               5.2e+09
  0.012500       182          3983        280.8              -1.10                   1991               6.3e+09
  0.010000       195          4700        283.4              -0.18                   1995               6.8e+09
  0.007500       216          5678        286.3               1.33                   1993               7.4e+09
  0.005000       242          7181        289.5               2.27                   3722               8.2e+09
  0.002500       300          9930        292.9               1.02                   3725               9.1e+09
  0.001250       360         13031        294.8               0.44                   3725               1.0e+10
  0.001000       384         14062        295.2               0.25                   3726               1.0e+10
  0.000750       414         15419        295.6               0.20                   3726               1.1e+10
  0.000500       444         17346        296.1               0.09                   3726               1.1e+10
  0.000125       567         24005        296.7              -0.01                   3722               1.3e+10
  0.000100       581         25054        296.7              -0.02                   3729               1.4e+10
  0.000050       645         28469        296.8              -0.00                   7185               1.4e+10
========================================================================================================================
""" 
import sys
from chemftr import df
from chemftr.molecule import load_casfile_to_pyscf

DE = 0.001  # max allowable phase error
CHI = 10    # number of bits for representation of coefficients
BETA = 16   # Value from paper 
THRESH_RANGE = [0.05, 0.025, 0.0125, 0.01, 0.0075, 0.005, 0.0025, 0.00125, 0.001, 0.00075, 0.0005, 0.000125, 0.0001, 0.00005] # various DF thresholds

# eri_reiher.h5 can be found at https://doi.org/10.5281/zenodo.4248322
REIHER_INTS = '../src/chemftr/integrals/eri_reiher.h5'  # path to integrals
reiher_mol, reiher_mf = load_casfile_to_pyscf(REIHER_INTS, num_alpha = 27, num_beta = 27)

# This writes out to file "double_factorization_reiher_femoco.txt"
df.double_factorization(reiher_mf, name='reiher_femoco', thresh_range=THRESH_RANGE, dE=DE, chi=CHI, beta=BETA)
