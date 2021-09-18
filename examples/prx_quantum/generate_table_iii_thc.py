""" Examples to show integral -> costing pipeline for SF and DF

Expected output:

[*] Reiher FeMoco Hamiltonian (cf. Table III)

[*] Li FeMoco Hamiltonian (cf. Table III)

"""
import os
import numpy as np
from importlib.resources import files
from chemftr import sf, df, thc
from chemftr.molecule import load_casfile_to_pyscf

""" Global defaults from paper """
DE = 0.001  # max allowable phase error
CHI = 10    # number of bits for representation of coefficients
REIHER_INTS = files('chemftr.integrals').joinpath('eri_reiher.h5')  # pre-packaged integrals
LI_INTS = files('chemftr.integrals').joinpath('eri_li.h5')  # pre-packaged integrals

if os.path.isfile(REIHER_INTS):

    """ Load Reiher FeMoco into memory """
    mol, mf = load_casfile_to_pyscf(REIHER_INTS, num_alpha = 27, num_beta = 27)
    n_orb = mf.mo_coeff.shape[0] * 2  # number spin orbitals is size of MOs x 2 for RHF

    """ THC factorization on Reiher FeMoco """
    BETA = 16
    NTHC = 350 
    _, thc_leaf, thc_central, info = thc.rank_reduce(mf._eri, NTHC)
    lam = thc.compute_lambda(mf, thc_leaf, thc_central)[0]
    
   # # Here we're using an initial calculation with a very rough estimate of the number of steps
   # # to give a more accurate number of steps. Then we input that into the function again.
    stps2 = thc.compute_cost(n_orb, lam, DE, chi=CHI, beta=BETA, M=NTHC, stps=20000)[0]
    thc_cost, thc_total_cost, thc_logical_qubits = thc.compute_cost(n_orb, lam, DE, chi=CHI, \
        beta=BETA, M=NTHC, stps=stps2)
    
    print("  [+] THC Factorization: ")
    print("      [-] Logical qubits: %s" % thc_logical_qubits)
    print("      [-] Toffoli count:  %.1e" % thc_total_cost)

if os.path.isfile(LI_INTS):
    """ Load Li FeMoco into memory """
    mol, mf = load_casfile_to_pyscf(LI_INTS,num_alpha = 74, num_beta = 39)
    n_orb = mf.mo_coeff.shape[0] * 2  # number spin orbitals is size of MOs x 2 for RHF

    """ THC factorization on Li FeMoco """
    BETA = 20 
    NTHC = 450 
    _, thc_leaf, thc_central, info = thc.rank_reduce(mf._eri, NTHC)
    lam = thc.compute_lambda(mf, thc_leaf, thc_central)[0]
    
   # # Here we're using an initial calculation with a very rough estimate of the number of steps
   # # to give a more accurate number of steps. Then we input that into the function again.
    stps2 = thc.compute_cost(n_orb, lam, DE, chi=CHI, beta=BETA, M=NTHC, stps=20000)[0]
    thc_cost, thc_total_cost, thc_logical_qubits = thc.compute_cost(n_orb, lam, DE, chi=CHI, \
        beta=BETA, M=NTHC, stps=stps2)
    
    print("  [+] THC Factorization: ")
    print("      [-] Logical qubits: %s" % thc_logical_qubits)
    print("      [-] Toffoli count:  %.1e" % thc_total_cost)
