""" Pretty-print a table comparing number of SF vectors retained versus accuracy and cost """

from pyscf import scf
from chemftr import thc 
from chemftr.molecule import rank_reduced_ccsd_t


def generate_costing_table(pyscf_mf,name='molecule',nthc_range=[250,300,350],dE=0.001,chi=10,beta=20,save_thc=False, use_kernel=True, no_triples=False, **kwargs):
    """ Print a table to file for testing how various THC thresholds impact cost, accuracy, etc.

    Args:
        pyscf_mf - PySCF mean field object
        name (str) - file will be saved to 'double_factorization_<name>.txt'
        nthc_range (list of ints) - list of number of THC vectors to retain in THC algorithm
        dE (float) - max allowable phase error (default: 0.001)
        chi (int) - number of bits for representation of coefficients (default: 10)
        beta (int) - not sure, but 20 was deemed sufficient for Li Hamiltonian (default: 20)
        save_thc (bool) - if True, save the THC factors (leaf and central only) 
        kwargs: additional keyword arguments to pass to thc.rank_reduce()
 
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
    escf, ecor, etot = rank_reduced_ccsd_t(pyscf_mf, eri_rr = None, use_kernel=use_kernel, no_triples=no_triples)

    exact_ecor = ecor
    exact_etot = etot

    filename = 'thc_factorization_'+name+'.txt'

    with open(filename,'w') as f:
        print("\n THC factorization data for '"+name+"'.",file=f) 
        print("    [*] using "+cas_info,file=f)                                          
        print("        [+]                      E(SCF): %18.8f" % escf,file=f) 
        if no_triples:
            print("        [+] Active space CCSD E(cor):    %18.8f" % ecor,file=f)
            print("        [+] Active space CCSD E(tot):    %18.8f" % etot,file=f)
        else:
            print("        [+] Active space CCSD(T) E(cor): %18.8f" % ecor,file=f)
            print("        [+] Active space CCSD(T) E(tot): %18.8f" % etot,file=f)
        print("{}".format('='*92),file=f)                                                                           
        if no_triples:
            print("{:^12} {:^24} {:^12} {:^20} {:^20}".format('M','CCSD error (mEh)','lambda', 'Toffoli count', 'logical qubits'),file=f)                             
        else:
            print("{:^12} {:^24} {:^12} {:^20} {:^20}".format('M','CCSD(T) error (mEh)','lambda', 'Toffoli count', 'logical qubits'),file=f)                             
        print("{}".format('-'*92),file=f)                                                                           
    for nthc in nthc_range:                                                                        
        # First, up: lambda and CCSD(T)
        if save_thc:
            fname = name + '_nTHC_' + str(nthc).zfill(5)  # will save as HDF5 and add .h5 extension
        else:
            fname = None
        eri_rr, thc_leaf, thc_central, info  = thc.rank_reduce(pyscf_mf._eri, nthc, thc_save_file=fname, **kwargs) 
        lam = thc.compute_lambda(pyscf_mf, thc_leaf, thc_central)[0]
        escf, ecor, etot = rank_reduced_ccsd_t(pyscf_mf, eri_rr, use_kernel=use_kernel, no_triples=no_triples)
        error = (etot - exact_etot)*1E3  # to mEh
      
        # now do costing
        stps1 = thc.compute_cost(num_spinorb, lam, DE, chi=CHI, beta=BETA, M=nthc, stps=20000)[0]
        thc_cost, thc_total_cost, thc_logical_qubits = thc.compute_cost(num_spinorb, lam, DE, chi=CHI, beta=BETA, M=nthc, stps=stps1)

        with open(filename,'a') as f:
            print("{:^12} {:^24.2f} {:^12.1f} {:^20.1e} {:^20}".format(nthc, error, lam, thc_total_cost, thc_logical_qubits),file=f)                                       
    with open(filename,'a') as f:
        print("{}".format('='*92),file=f)                                                                                  

    with open(filename, 'a') as f:
        print("THC factorization settings at exit:", file=f)
        for key, value in info.items():
            print("\t",key,value, file=f)
