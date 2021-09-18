""" Pretty-print a table comparing number of SF vectors retained versus accuracy and cost """

from pyscf import scf
from chemftr import df
from chemftr.molecule import rank_reduced_ccsd_t


def generate_costing_table(pyscf_mf,name='molecule',thresh_range=[0.0001],dE=0.001,chi=10,beta=20):
    """ Print a table to file for testing how various DF thresholds impact cost, accuracy, etc.

    Args:
        pyscf_mf - PySCF mean field object
        name (str) - file will be saved to 'double_factorization_<name>.txt'
        thresh_range (list of floats) - list of thresholds to try for DF algorithm
        dE (float) - max allowable phase error (default: 0.001)
        chi (int) - number of bits for representation of coefficients (default: 10)
        beta (int) - not sure, but 20 was deemed sufficient for Li Hamiltonian (default: 20)
 
    Returns:
       None
    """ 
                                                                                                     
    DE = dE  # max allowable phase error                                                              
    CHI = chi    # number of bits for representation of coefficients                                      
    BETA = beta   # not sure what we want here, but 20 was good enough for Li Hamiltonian

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
                                                                                                         
    # Reference calculation (eri_rr= None is full rank / exact ERIs)                                   
    escf, ecor, etot = rank_reduced_ccsd_t(pyscf_mf, eri_rr = None)

    exact_ecor = ecor
    exact_etot = etot

    filename = 'double_factorization_'+name+'.txt'

    with open(filename,'w') as f:
        print("\n Double low rank factorization data for '"+name+"'.",file=f) 
        print("    [*] using "+cas_info,file=f)                                          
        print("        [+]                      E(SCF): %18.8f" % escf,file=f) 
        print("        [+] Active space CCSD(T) E(cor): %18.8f" % ecor,file=f)
        print("        [+] Active space CCSD(T) E(tot): %18.8f" % etot,file=f)
        print("{}".format('='*120),file=f)                                                                           
        print("{:^12} {:^12} {:^12} {:^12} {:^24} {:^20} {:^20}".format('threshold','L','eigenvectors','lambda','CCSD(T) error (mEh)','logical qubits', 'Toffoli count'),file=f)                             
        print("{}".format('-'*120),file=f)                                                                           
    for thresh in thresh_range:                                                                        
        # First, up: lambda and CCSD(T)
        eri_rr, LR, L, Lxi = df.rank_reduce(pyscf_mf._eri, thresh=thresh) 
        lam = df.compute_lambda(pyscf_mf, LR)
        escf, ecor, etot = rank_reduced_ccsd_t(pyscf_mf, eri_rr)
        error = (etot - exact_etot)*1E3  # to mEh
      
        # now do costing
        stps1 = df.compute_cost(num_spinorb, lam, DE, L=L, Lxi=Lxi, chi=CHI, beta=BETA, stps=20000)[0]
        df_cost, df_total_cost, df_logical_qubits = df.compute_cost(num_spinorb, lam, DE, L=L, Lxi=Lxi, chi=CHI, beta=BETA, stps=stps1)

        with open(filename,'a') as f:
            print("{:^12.6f} {:^12} {:^12} {:^12.1f} {:^24.2f} {:^20} {:^20.1e}".format(thresh,L,Lxi,lam,error, df_logical_qubits, df_total_cost),file=f)                                       
    with open(filename,'a') as f:
        print("{}".format('='*120),file=f)                                                                                  
