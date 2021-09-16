""" Pretty-print a table comparing number of SF vectors retained versus accuracy and cost """

from pyscf import scf
from chemftr import sf
from chemftr.molecule import rank_reduced_ccsd_t


def single_factorization(pyscf_mf,name='molecule',rank_range=range(50,401,25),chi=10,dE=0.001):
    """ Print a table to file for testing how various SF ranks impact cost, accuracy, etc.

    Args:
        pyscf_mf - PySCF mean field object
        name (str) - file will be saved to 'double_factorization_<name>.txt'
        rank_range (list of ints) - list number of vectors to retain in SF algorithm
        dE (float) - max allowable phase error (default: 0.001)
        chi (int) - number of bits for representation of coefficients (default: 10)

    Returns:
       None
    """

    DE = dE  # max allowable phase error
    CHI = chi    # number of bits for representation of coefficients

    if isinstance(pyscf_mf, scf.rohf.ROHF):
        num_alpha, num_beta = pyscf_mf.nelec
        assert num_alpha + num_beta == pyscf_mf.mol.nelectron
    else:
        assert pyscf_mf.mol.nelectron % 2 == 0
        num_alpha = pyscf_mf.mol.nelectron // 2
        num_beta  = num_alpha

    num_orb = len(pyscf_mf.mo_coeff)
    num_spinorb = num_orb * 2

    cas_info = "CAS((%sa, %sb), %so)" % (num_alpha, num_beta, num_orb)

    # Reference calculation (eri_rr = None is full cholesky / exact ERIs)
    escf, ecor, etot = rank_reduced_ccsd_t(pyscf_mf, eri_rr = None)

    exact_ecor = ecor
    exact_etot = etot

    filename = 'single_factorization_'+name+'.txt'

    with open(filename,'w') as f:
        print("\n Single low rank factorization data for '"+name+"'.",file=f)
        print("    [*] using "+cas_info,file=f)
        print("        [+]                      E(SCF): %18.8f" % escf,file=f)
        print("        [+] Active space CCSD(T) E(cor): %18.8f" % ecor,file=f)
        print("        [+] Active space CCSD(T) E(tot): %18.8f" % etot,file=f)
        print("{}".format('='*89),file=f)
        print("{:^12} {:^12} {:^24} {:^20} {:^20}".format('L','lambda','CCSD(T) error (mEh)',\
            'logical qubits', 'Toffoli count'),file=f)
        print("{}".format('-'*89),file=f)
    for rank in rank_range:
        # First, up: lambda and CCSD(T)
        eri_rr, LR = sf.rank_reduce(pyscf_mf._eri, cholesky_dim=rank)
        lam = sf.compute_lambda(pyscf_mf, LR)
        escf, ecor, etot = rank_reduced_ccsd_t(pyscf_mf, eri_rr)
        error = (etot - exact_etot)*1E3  # to mEh

        # now do costing
        stps1 = sf.compute_cost(num_spinorb, lam, DE, L=rank, chi=CHI, stps=20000)[0]
        sf_cost, sf_total_cost, sf_logical_qubits = sf.compute_cost(num_spinorb, lam, DE, L=rank, chi=CHI,
                                                                stps=stps1)

        with open(filename,'a') as f:
            print("{:^12} {:^12.1f} {:^24.2f} {:^20} {:^20.1e}".format(rank,lam,error, sf_logical_qubits, sf_total_cost),file=f)                                       
    with open(filename,'a') as f:
        print("{}".format('='*89),file=f)                                                                                  

