""" AVAS pipeline """
import sys
import numpy as np
from pyscf import gto, scf, lo, tools, mcscf, ao2mo, cc
from pyscf.mcscf import avas
import h5py
from chemftr.util import gen_cas, RunSilent
from chemftr import sf, df
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial


class AVAS(object):

    def __init__(self,geometry,charge,multiplicity,basis='sto-3g',scf_type='rohf',symmetry='C1',name='pyscf',unit='angstrom',loc_type='pm'):
        if geometry.split('.')[-1].lower() == 'xyz':
            self.geometry = read_xyz(geometry)
        else:
            # assumes PySCF valid geometry string
            self.geometry = geometry

        self.charge = charge
        self.mult = multiplicity
        self.spin = multiplicity - 1
        self.basis = basis
        self.scf_type = scf_type.lower()
        self.symmetry = symmetry
        self.unit = unit

        self.name = name+'_{}_{}_mult{}'.format(self.scf_type, self.basis, self.mult)

        self.mol = gto.M(atom = self.geometry,
                basis = self.basis,
                spin = self.spin, # goes by 2 * S not 2S+1
                charge = self.charge,
                symmetry = self.symmetry,
                unit = self.unit)

        self.loc_type = loc_type
        self.scf_done = False
        self.localization_done = False

    def do_scf(self,stable=True,verbose=4):
        if self.scf_type == 'rohf':
            self.mf = scf.ROHF(self.mol)
        elif self.scf_type == 'rhf':
            self.mf = scf.RHF(self.mol)
        else:
            self.mf = scf.UHF(self.mol)

        self.mf.verbose = verbose
        self.mf.max_cycle = 200
        self.mf.damp = 0.2
        self.mf.chkfile = self.name+'.chk'
        self.mf.init_guess = 'chkfile'
        self.mf.conv_tol = 1e-8
        self.mf.kernel()

        if stable:
            # Will test for internal stability, re-run SCF with new orbitals, and test again
            mo1 = self.mf.stability()[0]
            dm1 = self.mf.make_rdm1(mo1, self.mf.mo_occ)
            self.mf = self.mf.run(dm1)
            self.mf.stability()

        self.scf_done = True


    def localize(self,loc_type='pm',checkfile=None,verbose=0,save=True):

        self.loc_type = loc_type

        if self.scf_done:
            print("Localization: using data from SCF in memory.")
            mol = self.mol
            scf_dict = {'e_tot'    : self.mf.e_tot,
                        'mo_energy': self.mf.mo_energy,
                        'mo_occ'   : self.mf.mo_occ,
                        'mo_coeff' : self.mf.mo_coeff}

        else:
            if checkfile is None:
                chkfile_path = self.name + '.chk'
            else: 
                chkfile_path = checkfile 
            print("Localization: using data from %s" % chkfile_path)
            try:
                mol, scf_dict = scf.chkfile.load_scf(chkfile_path)
            except FileNotFoundError:
                sys.exit("Inside "+sys._getframe().f_code.co_name+" \
                         \nCheckfile '"+str(chkfile_path)+"' not found. Did you generate it?\
                         \n Hint: you can pass checkpoint with the 'checkfile' keyword.")

        docc_idx = np.where(np.isclose(scf_dict['mo_occ'], 2.))[0]
        socc_idx = np.where(np.isclose(scf_dict['mo_occ'], 1.))[0]
        virt_idx = np.where(np.isclose(scf_dict['mo_occ'], 0.))[0]

        if loc_type == 'pm':
            print("Localizing doubly occupied ... ", end="")
            loc_docc_mo = lo.PM(mol, scf_dict['mo_coeff'][:, docc_idx]).kernel(verbose=verbose)
            print("singly occupied ... ", end="")
            loc_socc_mo = lo.PM(mol, scf_dict['mo_coeff'][:, socc_idx]).kernel(verbose=verbose)
            print("virtual ... ", end="")
            loc_virt_mo = lo.PM(mol, scf_dict['mo_coeff'][:, virt_idx]).kernel(verbose=verbose)
            print("DONE")

            if self.scf_done:
                self.mf.mo_coeff[:,docc_idx] = loc_docc_mo.copy()
                self.mf.mo_coeff[:,socc_idx] = loc_socc_mo.copy()
                self.mf.mo_coeff[:,virt_idx] = loc_virt_mo.copy()

        elif loc_type == 'er':
            print("Localizing doubly occupied ... ", end="")
            loc_docc_mo = lo.ER(mol, scf_dict['mo_coeff'][:, docc_idx]).kernel(verbose=verbose)
            print("singly occupied ... ", end="")
            loc_socc_mo = lo.ER(mol, scf_dict['mo_coeff'][:, socc_idx]).kernel(verbose=verbose)
            print("virtual ... ", end="")
            loc_virt_mo = lo.ER(mol, scf_dict['mo_coeff'][:, virt_idx]).kernel(verbose=verbose)
            print("DONE")

            if self.scf_done:
                self.mf.mo_coeff[:,docc_idx] = loc_docc_mo.copy()
                self.mf.mo_coeff[:,socc_idx] = loc_socc_mo.copy()
                self.mf.mo_coeff[:,virt_idx] = loc_virt_mo.copy()

        self.localization_done = True

        if save:
            loc_mo_coeff = np.hstack((loc_docc_mo, loc_socc_mo, loc_virt_mo))
            np.save(self.name+"_{}_localized_mocoeffs".format(self.loc_type), loc_mo_coeff)


            localized_chkfile_name = self.name +'_{}_localized.chk'.format(self.loc_type)
            scf.chkfile.dump_scf(mol, localized_chkfile_name, scf_dict['e_tot'], scf_dict['mo_energy'], loc_mo_coeff, scf_dict['mo_occ'])

            molden_filename = self.name+'_{}_localized.molden'.format(self.loc_type)
            tools.molden.from_chkfile(molden_filename, localized_chkfile_name)


    def do_avas(self,ao_list,checkfile=None,**kwargs):

        if self.scf_done:
            print("AVAS: using data from SCF in memory.")
            scf_dict = {'e_tot'    : self.mf.e_tot,
                        'mo_energy': self.mf.mo_energy,
                        'mo_occ'   : self.mf.mo_occ,
                        'mo_coeff' : self.mf.mo_coeff}
            print("Original number of orbitals ", self.mf.mo_coeff.shape[0])
        else:

            if checkfile is None:
                chkfile_path = self.name +'_{}_localized.chk'.format(self.loc_type) 
            else: 
                chkfile_path = checkfile 
            print("AVAS: using data from %s" % chkfile_path)
            try:
                self.mol, scf_dict = scf.chkfile.load_scf(chkfile_path)
            except FileNotFoundError:
                sys.exit("Inside "+sys._getframe().f_code.co_name+" \
                         \nCheckfile '"+str(chkfile_path)+"' not found. Did you generate it?\
                         \n Hint: you can pass checkpoint with the 'checkfile' keyword.")

            self.mf = scf.ROHF(self.mol)
            self.mf.e_tot = scf_dict['e_tot']
            self.mf.mo_coeff = scf_dict['mo_coeff']
            self.mf.mo_occ = scf_dict['mo_occ']
            self.mf.mo_energy = scf_dict['mo_energy']
            print("Original number of orbitals ", self.mf.mo_coeff.shape[0])

        print("({}, {}) high spin config (alpha, beta)".format(self.mf.nelec[0], self.mf.nelec[1]))

        ao_labels = self.mol.ao_labels()
        #print(ao_labels)
        # Note: requires openshell_option = 3 for this to work
        avas_output = avas.avas(self.mf, ao_list, canonicalize=False, openshell_option=3,**kwargs)
        self.active_norb, self.active_ne, self.reordered_orbs = avas_output
        print("Active Orb:      ", self.active_norb)
        print("Active Ele:      ", self.active_ne)
        print("Reordered shape: ", self.reordered_orbs.shape)

        # generate the new active space hamiltonian using mo coeffs from avas
        h1_avas, eri_avas, ecore, active_alpha, active_beta = gen_cas(self.mf, self.active_norb,
            self.active_ne, avas_orbs=self.reordered_orbs)
        print(active_alpha, active_beta)

        # save set of localized orbitals for active space
        frozen_alpha = self.mf.nelec[0] - active_alpha

        active_space_idx = slice(frozen_alpha,frozen_alpha+self.active_norb)
        active_mos = self.reordered_orbs[:,active_space_idx]
        tools.molden.from_mo(self.mf.mol, 'avas_localized_orbitals_'+self.name+'.molden', mo_coeff=active_mos)

        # verify using OpenFermion
        self.verify_using_openfermion(self.mf, h1_avas, eri_avas, ecore, active_alpha, active_beta)

        h5name = "avas_hamiltonian_"+self.name+".h5"
        self.save_cas(h5name, h1_avas, eri_avas, ecore, active_alpha, active_beta)

    def do_single_factorization(self,rank_range=range(50,401,25),chi=10,dE=0.001,use_kernel=True,\
        integral_path=None):
                                                                                                         
        DE = dE  # max allowable phase error                                                              
        CHI = chi    # number of bits for representation of coefficients                                      
        USE_KERNEL = use_kernel # do re-run SCF prior to CCSD_T?                                                  
      
        if integral_path is None:
            INTS = 'avas_hamiltonian_'+self.name+'.h5' 
        else:
            INTS = integral_path
    
        try:
        # Get active space info
            with h5py.File(INTS, "r") as f:
                num_alpha = int(f['active_nalpha'][()]) 
                num_beta  = int(f['active_nbeta'][()])
                try:
                    num_orb   = np.asarray(f['h0'][()]).shape[0]
                except KeyError:
                    num_orb = np.asarray(f['h1'][()]).shape[0]

        except FileNotFoundError:
            sys.exit("Inside "+sys._getframe().f_code.co_name+" \
                     \nAVAS Hamiltonian '"+str(INTS)+"' not found. Did you generate it?\
                     \n Hint: you can pass Hamiltonian with the 'integral_path' keyword.")
        
        cas_info = "CAS((%sa, %sb), %so)" % (num_alpha, num_beta, num_orb)

        # Reference calculation (dim = None is full cholesky / exact ERIs)                                   
        # run silently                                                                                       
        #with RunSilent():
        escf, ecor, etot = sf.compute_ccsd_t(cholesky_dim=None,integral_path=INTS,use_kernel=USE_KERNEL)
                                                                                                             
        exact_ecor = ecor
        exact_etot = etot
    
        filename = 'single_factorization_'+self.name+'.txt'
    
        with open(filename,'w') as f:
            print("\n Single low rank factorization data for '"+self.name+"'.",file=f)                                          
            print("    [*] using "+cas_info,file=f)                                          
            print("        [+]                      E(SCF): %18.8f" % escf,file=f) 
            print("        [+] Active space CCSD(T) E(cor): %18.8f" % ecor,file=f)
            print("        [+] Active space CCSD(T) E(tot): %18.8f" % etot,file=f)
            print("{}".format('='*89),file=f)                                                                           
            print("{:^12} {:^12} {:^24} {:^20} {:^20}".format('L','lambda','CCSD(T) error (mEh)','logical qubits', 'Toffoli count'),file=f)                             
            print("{}".format('-'*89),file=f)                                                                           
        for rank in rank_range:                                                                        
            # run silently                                                                                   
            # with RunSilent():
    
            # First, up: lambda and CCSD(T)
            n_orb, lam = sf.compute_lambda(cholesky_dim=rank, integral_path=INTS)
            escf, ecor, etot = sf.compute_ccsd_t(cholesky_dim=rank, integral_path=INTS, use_kernel=USE_KERNEL)
            error = (etot - exact_etot)*1E3  # to mEh
          
            # now do costing
            stps1 = sf.compute_cost(n_orb, lam, DE, L=rank, chi=CHI, stps=20000)[0]
            sf_cost, sf_total_cost, sf_logical_qubits = sf.compute_cost(n_orb, lam, DE, L=rank, chi=CHI,
                                                                    stps=stps1)

            with open(filename,'a') as f:
                print("{:^12} {:^12.1f} {:^24.2f} {:^20} {:^20.1e}".format(rank,lam,error, sf_logical_qubits, sf_total_cost),file=f)                                       
        with open(filename,'a') as f:
            print("{}".format('='*89),file=f)                                                                                  

    def do_double_factorization(self,thresh_range,dE=0.001,chi=10,beta=20,use_kernel=True,
        integral_path=None):
                                                                                                         
        DE = dE  # max allowable phase error                                                              
        CHI = chi    # number of bits for representation of coefficients                                      
        BETA = beta   # not sure what we want here, but 20 was good enough for Li Hamiltonian
        USE_KERNEL = use_kernel # do re-run SCF prior to CCSD_T?                                                  

        if integral_path is None:
            INTS = 'avas_hamiltonian_'+self.name+'.h5' 
        else:
            INTS = integral_path
   
        try: 
            # Get active space info
            with h5py.File(INTS, "r") as f:
                num_alpha = int(f['active_nalpha'][()]) 
                num_beta  = int(f['active_nbeta'][()])
                try:
                    num_orb   = np.asarray(f['h0'][()]).shape[0]
                except KeyError:
                    num_orb  = np.asarray(f['h1'][()]).shape[0]
        except FileNotFoundError:
            sys.exit("Inside "+sys._getframe().f_code.co_name+" \
                     \nAVAS Hamiltonian '"+str(INTS)+"' not found. Did you generate it?\
                     \n Hint: you can pass Hamiltonian with the 'integral_path' keyword.")

   
        
        cas_info = "CAS((%sa, %sb), %so)" % (num_alpha, num_beta, num_orb)
                                                                                                             
        # Reference calculation (dim = None is full cholesky / exact ERIs)                                   
        # run silently                                                                                       
        with RunSilent():
            escf, ecor, etot = df.compute_ccsd_t(thresh=0.0,integral_path=INTS,use_kernel=USE_KERNEL)                 
                                                                                                             
        exact_ecor = ecor
        exact_etot = etot
    
        filename = 'double_factorization_'+self.name+'.txt'
    
        with open(filename,'w') as f:
            print("\n Double low rank factorization data for '"+self.name+"'.",file=f) 
            print("    [*] using "+cas_info,file=f)                                          
            print("        [+]                      E(SCF): %18.8f" % escf,file=f) 
            print("        [+] Active space CCSD(T) E(cor): %18.8f" % ecor,file=f)
            print("        [+] Active space CCSD(T) E(tot): %18.8f" % etot,file=f)
            print("{}".format('='*120),file=f)                                                                           
            print("{:^12} {:^12} {:^12} {:^12} {:^24} {:^20} {:^20}".format('threshold','L','eigenvectors','lambda','CCSD(T) error (mEh)','logical qubits', 'Toffoli count'),file=f)                             
            print("{}".format('-'*120),file=f)                                                                           
        for thresh in thresh_range:                                                                        
            # run silently                                                                                   
            # with RunSilent():
            # First, up: lambda and CCSD(T)
            n_orb, lam, L, Lxi = df.compute_lambda(thresh, integral_path=INTS)
            escf, ecor, etot = df.compute_ccsd_t(thresh, integral_path=INTS, use_kernel=USE_KERNEL)
            error = (etot - exact_etot)*1E3  # to mEh
          
            # now do costing
            stps1 = df.compute_cost(n_orb, lam, DE, L=L, Lxi=Lxi, chi=CHI, beta=BETA, stps=20000)[0]
            df_cost, df_total_cost, df_logical_qubits = df.compute_cost(n_orb, lam, DE, L=L, Lxi=Lxi, chi=CHI, beta=BETA, stps=stps1)
    
            with open(filename,'a') as f:
                print("{:^12.6f} {:^12} {:^12} {:^12.1f} {:^24.2f} {:^20} {:^20.1e}".format(thresh,L,Lxi,lam,error, df_logical_qubits, df_total_cost),file=f)                                       
        with open(filename,'a') as f:
            print("{}".format('='*120),file=f)                                                                                  

    @staticmethod
    def save_cas(fname,h1,eri,ecore,num_alpha,num_beta):
        with h5py.File(fname, 'w') as fid:
            fid.create_dataset('ecore', data=float(ecore), dtype=float)
            fid.create_dataset('h0', data=h1)  # note the name change for consistency with THC paper
            fid.create_dataset('eri', data=eri)
            fid.create_dataset('active_nalpha', data=int(num_alpha), dtype=int)
            fid.create_dataset('active_nbeta', data=int(num_beta), dtype=int)

    @staticmethod
    def verify_using_openfermion(mf, h1, eri, ecore, active_alpha, active_beta):
        # First, create OpenFermion Hamiltonian from one- and two-body integrals + scalar constant
        # See PQRS convention in OpenFermion.hamiltonians._molecular_data
        # h[p,q,r,s] = (ps|qr)
        tei = np.asarray(
            eri.transpose(0, 2, 3, 1), order='C')
        soei, stei = spinorb_from_spatial(h1, tei)
        astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
        active_space_ham = of.InteractionOperator(ecore, soei, 0.25 * astei)


        active_norb = h1.shape[0]
        alpha_diag = [1] * active_alpha + [0] * (active_norb - active_alpha)
        beta_diag = [1] * active_beta + [0] * (active_norb - active_beta)
        aspace_spin_opdm = alpha_diag + beta_diag
        aspace_spin_opdm[::2] = alpha_diag
        aspace_spin_opdm[1::2] = beta_diag
        aspace_spin_opdm = np.diag(aspace_spin_opdm)
        aspace_spin_tpdm = 2 * of.wedge(aspace_spin_opdm, aspace_spin_opdm, (1, 1), (1, 1))
        aspace_rdms = of.InteractionRDM(aspace_spin_opdm, aspace_spin_tpdm)
        print("E(SCF), OF vs MF: %12.6f, %12.6f" % (aspace_rdms.expectation(active_space_ham).real, mf.e_tot))
        assert np.isclose(aspace_rdms.expectation(active_space_ham).real, mf.e_tot)

def read_xyz(filename):
    geometry = []
    with open(filename,'r') as f:
        lines = f.readlines()
        n_atoms = int(lines[0])
        for line in lines:
            try:
                atom = str(line.split()[0])
                assert len(atom) <= 2
                xcor = float(line.split()[1])
                ycor = float(line.split()[2])
                zcor = float(line.split()[3])
                geometry.append(line)
            except:
                continue
    assert len(geometry) == n_atoms
    return geometry

if __name__ == '__main__':

    chem = AVAS('aligned_xy.xyz',charge=0,multiplicity=6,basis='sto3g')
    chem.do_scf()
    chem.localize(loc_type='pm')
    chem.do_avas()

    # make pretty SF costing table
    chem.do_single_factorization()

    # make pretty DF costing table
    chem.do_double_factorization()
