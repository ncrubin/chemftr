""" Generate Table XI from THC costing paper

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
from chemftr import sf
from chemftr.util import RunSilent
from chemftr.molecule import load_casfile_to_pyscf, rank_reduced_ccsd_t

DE = 0.001  # max allowable phase error
CHI = 10    # number of bits for representation of coefficients
USE_KERNEL = True # do re-run SCF prior to CCSD_T?

# eri_reiher.h5 can be found at https://doi.org/10.5281/zenodo.4248322
REIHER_INTS = '../src/chemftr/integrals/eri_reiher.h5'  # path to integrals
reiher_mol, reiher_mf = load_casfile_to_pyscf(REIHER_INTS, num_alpha = 27, num_beta = 27)
n_orb = reiher_mf.mo_coeff.shape[0] * 2  # number spin orbitals is number of MOs x 2 in RHF

# Reference calculation (dim = None is full cholesky / exact ERIs)
# run silently
with RunSilent():
    escf, ecor, etot = rank_reduced_ccsd_t(reiher_mf, eri_rr = None, use_kernel = USE_KERNEL)

exact_ecor = ecor

print("\nTable XI. Single low rank factorization data for") 
print("          the Reiher Hamiltonian.") 
print("{}".format('='*48))
print("{:^12} {:^12} {:^24}".format('L','lambda','CCSD(T) error (mEh)'))
print("{}".format('-'*48))
for rank in range(50,401,25):
    # run silently
    with RunSilent():
        eri_rr, sf_factors = sf.rank_reduce(reiher_mf._eri, rank)
        lam = sf.compute_lambda(reiher_mf, sf_factors)
        escf, ecor, etot   = rank_reduced_ccsd_t(reiher_mf, eri_rr) 
        error = (ecor - exact_ecor)*1E3  # to mEh
    print("{:^12} {:^12.1f} {:^24.2f}".format(rank,lam,error))
print("{}".format('='*48))
