""" Generate Table XIII from THC costing paper

Expected output:

Table XIII. Double low rank factorization data for
          the Reiher Hamiltonian.
============================================================================
 threshold        L       eigenvectors    lambda      CCSD(T) error (mEh)
----------------------------------------------------------------------------
  0.100000        53          765         253.6              -87.91
  0.050000        95          1274        262.1              -12.39
  0.025000       135          2319        272.0              -2.78
  0.012500       182          3983        280.8              -1.10
  0.010000       195          4700        283.4              -0.18
  0.007500       216          5678        286.3               1.33
  0.005000       242          7181        289.5               2.27
  0.002500       300          9930        292.9               1.02
  0.001250       360         13031        294.8               0.44
  0.001000       384         14062        295.2               0.25
  0.000750       414         15419        295.6               0.20
  0.000500       444         17346        296.1               0.09
  0.000125       567         24005        296.7              -0.01
  0.000100       581         25054        296.7              -0.02
  0.000050       645         28469        296.8              -0.00
============================================================================
"""
try:                                                                                                 
    from importlib.resources import files                                                            
except ImportError:                                                                                  
    from importlib_resources import files                                                            
from chemftr import df
from chemftr.utils import RunSilent
from chemftr.molecule import load_casfile_to_pyscf, rank_reduced_ccsd_t

DE = 0.001  # max allowable phase error
CHI = 10    # number of bits for representation of coefficients
USE_KERNEL = True  # Re-do SCF prior to CCSD(T)? 

# eri_reiher.h5 can be found at https://doi.org/10.5281/zenodo.4248322
REIHER_INTS = files('chemftr.integrals').joinpath('eri_reiher.h5')  # pre-packaged integrals
reiher_mol, reiher_mf = load_casfile_to_pyscf(REIHER_INTS, num_alpha = 27, num_beta = 27)
n_orb = reiher_mf.mo_coeff.shape[0]  # num spin orbitals is num MOs x 2 for RHF

# Reference calculation (dim = None is full cholesky / exact ERIs)
# run silently
with RunSilent():
    escf, ecor, etot = rank_reduced_ccsd_t(reiher_mf, eri_rr = None, use_kernel = USE_KERNEL)

exact_ecor = ecor

print("\nTable XIII. Double low rank factorization data for")
print("          the Reiher Hamiltonian.")
print("{}".format('='*76))
print("{:^12} {:^12} {:^12} {:^12} {:^24}".format('threshold','L', 'eigenvectors','lambda',
                                                  'CCSD(T) error (mEh)'))
print("{}".format('-'*76))
for thresh in [0.1, 0.05, 0.025, 0.0125, 0.01, 0.0075, 0.005, 0.0025, 0.00125, 0.001, 0.00075, \
               0.0005, 0.000125, 0.0001, 0.00005]:
    with RunSilent():
        eri_rr, df_factors, L, Lxi = df.rank_reduce(reiher_mf._eri, thresh)
        lam  = df.compute_lambda(reiher_mf, df_factors)
        escf, ecor, etot = rank_reduced_ccsd_t(reiher_mf, eri_rr) 
        error = (ecor - exact_ecor)*1E3  # to mEh
    print("{:^12.6f} {:^12} {:^12} {:^12.1f} {:^24.2f}".format(thresh,L,Lxi,lam,error))
print("{}".format('='*76))
