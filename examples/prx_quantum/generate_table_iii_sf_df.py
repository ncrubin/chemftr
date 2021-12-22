""" Examples to show integral -> costing pipeline for SF and DF

Expected output:

[*] Reiher FeMoco Hamiltonian (cf. Table III)
  [+] Single Factorization:
      [-] Logical qubits: 3320
      [-] Toffoli count:  9.5e+10
  [+] Double Factorization:
      [-] Logical qubits: 3725
      [-] Toffoli count:  1.0e+10

[*] Li FeMoco Hamiltonian (cf. Table III)
  [+] Single Factorization:
      [-] Logical qubits: 3628
      [-] Toffoli count:  1.2e+11
  [+] Double Factorization:
      [-] Logical qubits: 6404
      [-] Toffoli count:  6.4e+10
"""
import os
import numpy as np
try:                                                                                                 
    from importlib.resources import files                                                            
except ImportError:                                                                                  
    from importlib_resources import files                                                            
from chemftr import sf, df
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

    """ Single factorization on Reiher FeMoco """
    RANK = 200
    sf_factors = sf.factorize(mf._eri, rank=RANK)[1]
    lam = sf.compute_lambda(mf, sf_factors)  
    
    # Here we're using an initial calculation with a very rough estimate of the number of steps
    # to give a more accurate number of steps. Then we input that into the function again.
    stps1 = sf.compute_cost(n_orb, lam, DE, L=RANK, chi=CHI,stps=20000)[0]
    sf_cost, sf_total_cost, sf_logical_qubits = sf.compute_cost(n_orb, lam, DE, L=RANK,
        chi=CHI,stps=stps1)
    
    print("[*] Reiher FeMoco Hamiltonian (cf. Table III)")
    print("  [+] Single Factorization: ")
    print("      [-] Logical qubits: %s" % sf_logical_qubits)
    print("      [-] Toffoli count:  %.1e" % sf_total_cost)
    assert sf_logical_qubits == 3320 
    assert '{:.1e}'.format(sf_total_cost) == '9.5e+10'
    
    """ Double factorization on Reiher FeMoco """
    BETA = 16
    THRESH = 0.00125
    _, df_factors, rank, num_eigen = df.factorize(mf._eri, thresh=THRESH)
    lam = df.compute_lambda(mf, df_factors)
    
    # Here we're using an initial calculation with a very rough estimate of the number of steps
    # to give a more accurate number of steps. Then we input that into the function again.
    stps2 = df.compute_cost(n_orb, lam, DE, L=rank, Lxi=num_eigen, chi=CHI, beta=BETA,stps=20000)[0]
    df_cost, df_total_cost, df_logical_qubits = df.compute_cost(n_orb, lam, DE, L=rank, Lxi=num_eigen,
         chi=CHI, beta=BETA,stps=stps2)
    
    print("  [+] Double Factorization: ")
    print("      [-] Logical qubits: %s" % df_logical_qubits)
    print("      [-] Toffoli count:  %.1e" % df_total_cost)
    assert df_logical_qubits == 3725
    assert '{:.1e}'.format(df_total_cost) == '1.0e+10'

if os.path.isfile(LI_INTS):
    """ Load Li FeMoco into memory """
    mol, mf = load_casfile_to_pyscf(LI_INTS,num_alpha = 74, num_beta = 39)
    n_orb = mf.mo_coeff.shape[0] * 2  # number spin orbitals is size of MOs x 2 for RHF

    """ Single factorization on Li FeMoco """
    RANK = 275
    sf_factors = sf.factorize(mf._eri, rank=RANK)[1]
    lam = sf.compute_lambda(mf, sf_factors)  
    
    # Here we're using an initial calculation with a very rough estimate of the number of steps
    # to give a more accurate number of steps. Then we input that into the function again.
    stps1 = sf.compute_cost(n_orb, lam, DE, L=RANK, chi=CHI,stps=20000)[0]
    sf_cost, sf_total_cost, sf_logical_qubits = sf.compute_cost(n_orb, lam, DE, L=RANK,
        chi=CHI,stps=stps1)
    
    print("\n[*] Li FeMoco Hamiltonian (cf. Table III)")
    print("  [+] Single Factorization: ")
    print("      [-] Logical qubits: %s" % sf_logical_qubits)
    print("      [-] Toffoli count:  %.1e" % sf_total_cost)
    assert sf_logical_qubits == 3628 
    assert '{:.1e}'.format(sf_total_cost) == '1.2e+11'
    
    """ Double factorization on Li FeMoco """
    BETA = 20
    THRESH = 0.00125
    _, df_factors, rank, num_eigen = df.factorize(mf._eri, thresh=THRESH)
    lam = df.compute_lambda(mf, df_factors)

    # Here we're using an initial calculation with a very rough estimate of the number of steps
    # to give a more accurate number of steps. Then we input that into the function again.
    stps2 = df.compute_cost(n_orb, lam, DE, L=rank, Lxi=num_eigen, chi=CHI, beta=BETA,stps=20000)[0]
    df_cost, df_total_cost, df_logical_qubits = df.compute_cost(n_orb, lam, DE, L=rank, Lxi=num_eigen,
         chi=CHI, beta=BETA,stps=stps2)
    
    print("  [+] Double Factorization: ")
    print("      [-] Logical qubits: %s" % df_logical_qubits)
    print("      [-] Toffoli count:  %.1e" % df_total_cost)
    assert df_logical_qubits == 6404
    assert '{:.1e}'.format(df_total_cost) == '6.4e+10'
