""" Examples to show integral -> costing pipeline for SF and DF"""
from chemftr import sf, df
import numpy as np

""" Global defaults from paper """
DE = 0.001  # max allowable phase error
CHI = 10    # number of bits for representation of coefficients
REIHER_INTS = './src/chemftr/integrals/eri_reiher.h5'  # path to integrals

""" Single factorization on Reiher FeMoco """
RANK = 200
n_orb, lam = sf.compute_lambda(cholesky_dim=RANK, integral_path=REIHER_INTS)

# Here we're using an initial calculation with a very rough estimate of the number of steps
# to give a more accurate number of steps. Then we input that into the function again.
stps1 = sf.compute_cost(n_orb, lam, DE, L=RANK, chi=CHI,stps=20000)[0]
sf_cost, sf_total_cost, sf_logical_qubits = sf.compute_cost(n_orb, lam, DE, L=RANK,
    chi=CHI,stps=stps1)

print("[*] Reiher FeMoco Hamiltonian (cf. Table III)")
print("  [+] Single Factorization: ")
print("      [-] Logical qubits: %s" % sf_logical_qubits)
print("      [-] Toffoli count:  %.1e" % sf_total_cost)

""" Double factorization on Reiher FeMoco """
BETA = 16
THRESH = 0.00125
n_orb, lam, rank, num_eigen = df.compute_lambda(thresh=THRESH, integral_path=REIHER_INTS)

# Here we're using an initial calculation with a very rough estimate of the number of steps
# to give a more accurate number of steps. Then we input that into the function again.
stps2 = df.compute_cost(n_orb, lam, DE, L=rank, Lxi=num_eigen, chi=CHI, beta=BETA,stps=20000)[0]
df_cost, df_total_cost, df_logical_qubits = df.compute_cost(n_orb, lam, DE, L=rank, Lxi=num_eigen,
     chi=CHI, beta=BETA,stps=stps2)

print("  [+] Double Factorization: ")
print("      [-] Logical qubits: %s" % df_logical_qubits)
print("      [-] Toffoli count:  %.1e" % df_total_cost)

""" Example to reproduce first two columns of Table XI in the THC paper"""

print("\n[*] Reiher FeMoco Hamiltonian (cf. Table XI)")
print("        Rank (L)          Lambda            ")
print("============================================")
for rank in range(50,401,25):
    n_orb, lam = sf.compute_lambda(cholesky_dim=rank, integral_path=REIHER_INTS)
    print("            %4d          %4.1f             " % (rank, lam))