""" Generate Table XI from THC costing paper including cost estimates

Expected output:

Table XI. Single low rank factorization data for
          the Reiher Hamiltonian.
================================================
     L          lambda      CCSD(T) error (mEh)
------------------------------------------------
     50         3367.6              4.79
     75         3657.9              0.21
    100         3854.3              1.55
    125         3997.4              3.08
    150         4112.7              2.07
    175         4199.2              1.63
    200         4258.0              0.10
    225         4300.7              0.38
    250         4331.9              0.16
    275         4354.9              0.29
    300         4372.0              0.18
    325         4385.3              0.13
    350         4395.6              0.06
    375         4403.4              0.03
    400         4409.3              0.02
================================================
""" 
import sys
from chemftr import df
from chemftr.molecule import load_casfile_to_pyscf

DE = 0.001  # max allowable phase error
CHI = 10    # number of bits for representation of coefficients
BETA = 16   # Value from paper 
THRESH_RANGE = [0.05, 0.025, 0.0125, 0.01, 0.0075, 0.005, 0.0025, 0.00125, 0.001, 0.00075, 0.0005, 0.000125, 0.0001, 0.00005] # various DF thresholds
REIHER_INTS = '../src/chemftr/integrals/eri_reiher.h5'  # path to integrals
reiher_mol, reiher_mf = load_casfile_to_pyscf(REIHER_INTS, num_alpha = 27, num_beta = 27)

# This writes out to file "double_factorization_reiher_femoco.txt"
df.double_factorization(reiher_mf, name='reiher_femoco', thresh_range=THRESH_RANGE, dE=DE, chi=CHI, beta=BETA)
