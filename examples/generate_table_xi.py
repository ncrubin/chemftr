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
from io import StringIO

class NullIO(StringIO):
    """ Class to replace sys.stdout with silence """
    def write(self, txt):
        pass

DE = 0.001  # max allowable phase error
CHI = 10    # number of bits for representation of coefficients
USE_KERNEL = False # do re-run SCF prior to CCSD_T?
REIHER_INTS = '../src/chemftr/integrals/eri_reiher.h5'  # path to integrals
#REIHER_INTS = '../src/chemftr/integrals/eri_reiher_newscf.h5'  # path to integrals

# Reference calculation (dim = None is full cholesky / exact ERIs)
# run silently
sys.stdout = NullIO()
escf, ecor, etot = sf.compute_ccsd_t(cholesky_dim=None,integral_path=REIHER_INTS,\
                                     num_alpha=27,num_beta=27,use_kernel=USE_KERNEL)
sys.stdout = sys.__stdout__

exact_ecor = ecor

print("\nTable XI. Single low rank factorization data for") 
print("          the Reiher Hamiltonian.") 
print("{}".format('='*48))
print("{:^12} {:^12} {:^24}".format('L','lambda','CCSD(T) error (mEh)'))
print("{}".format('-'*48))
for rank in range(50,401,25):
    # run silently
    sys.stdout = NullIO()
    n_orb, lam = sf.compute_lambda(cholesky_dim=rank, integral_path=REIHER_INTS)
    escf, ecor, etot = sf.compute_ccsd_t(cholesky_dim=rank, integral_path=REIHER_INTS,
                                         num_alpha=27,num_beta=27,  use_kernel=USE_KERNEL)
    error = (ecor - exact_ecor)*1E3  # to mEh
    sys.stdout = sys.__stdout__
    print("{:^12} {:^12.1f} {:^24.2f}".format(rank,lam,error))
print("{}".format('='*48))
