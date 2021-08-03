""" AVAS pipeline """
import numpy as np
from pyscf import gto, scf, lo, tools, mcscf, ao2mo, cc
from pyscf.mcscf import avas
import h5py
from chemftr.util import gen_cas
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial


class AVAS(object):

    def __init__(self,geometry,charge,multiplicity,basis='sto-3g',scf_type='rohf',symmetry='C1',name='pyscf',unit='angstrom'):
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
        self.mf.conv_tol = 1e-10
        self.mf.kernel()

        if stable:
            # Will test for internal stability, re-run SCF with new orbitals, and test again
            mo1 = self.mf.stability()[0]
            dm1 = self.mf.make_rdm1(mo1, self.mf.mo_occ)
            self.mf = self.mf.run(dm1)
            self.mf.stability()

        self.scf_done = True


    def localize(self,loc_type='pm',verbose=0,save=True):

        if self.scf_done:
            print("Localization: using data from SCF in memory.")
            mol = self.mol
            scf_dict = {'e_tot'    : self.mf.e_tot,
                        'mo_energy': self.mf.mo_energy,
                        'mo_occ'   : self.mf.mo_occ,
                        'mo_coeff' : self.mf.mo_coeff}

        else:
            chkfile_path = self.name + '.chk'
            print("Localization: using data from %s" % chkfile_path)
            mol, scf_dict = scf.chkfile.load_scf(chkfile_path)

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

            self.mf.mo_coeff[:,docc_idx] = loc_docc_mo.copy()
            self.mf.mo_coeff[:,socc_idx] = loc_socc_mo.copy()
            self.mf.mo_coeff[:,virt_idx] = loc_virt_mo.copy()

        self.localization_done = True

        if save:
            loc_mo_coeff = np.hstack((loc_docc_mo, loc_socc_mo, loc_virt_mo))
            np.save(self.name+"_{}_localized_mocoeffs".format(loc_type), loc_mo_coeff)


            localized_chkfile_name = self.name +'_{}_localized.chk'.format(loc_type)
            scf.chkfile.dump_scf(mol, localized_chkfile_name, scf_dict['e_tot'], scf_dict['mo_energy'], loc_mo_coeff, scf_dict['mo_occ'])

            molden_filename = self.name+'_{}_localized.molden'.format(loc_type)
            tools.molden.from_chkfile(molden_filename, localized_chkfile_name)


    def do_avas(self,ao_list=['C 2pz','N 2pz', 'S 3s', 'S 3p', 'S 3s', 'Fe 3d']):

        if self.scf_done:
            print("AVAS: using data from SCF in memory.")
            scf_dict = {'e_tot'    : self.mf.e_tot,
                        'mo_energy': self.mf.mo_energy,
                        'mo_occ'   : self.mf.mo_occ,
                        'mo_coeff' : self.mf.mo_coeff}
            print("Original number of orbitals ", self.mf.mo_coeff.shape[0])
        else:
            chkfile_path = self.name + '.chk'
            print("AVAS: using data from %s" % chkfile_path)
            self.mol, scf_dict = scf.chkfile.load_scf(chkfile_path)

            self.mf = scf.ROHF(self.mol)
            self.mf.e_tot = scf_dict['e_tot']
            self.mf.mo_coeff = scf_dict['mo_coeff']
            self.mf.mo_occ = scf_dict['mo_occ']
            self.mf.mo_energy = scf_dict['mo_energy']
            print("Original number of orbitals ", self.mf.mo_coeff.shape[0])

            #d0 = self.mf.make_rdm1()
            #self.mf.kernel(d0)

        print("({}, {}) high spin config (alpha, beta)".format(self.mf.nelec[0], self.mf.nelec[1]))

        ao_labels = self.mol.ao_labels()
        #print(ao_labels)
        # Note: requires openshell_option = 3 for this to work
        avas_output = avas.avas(self.mf, ao_list, canonicalize=False, openshell_option=3)
        print(avas_output)
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

    chem = AVAS('aligned_xy.xyz',charge=3,multiplicity=6,basis='sto3g')
    chem.do_scf()
    chem.localize(loc_type='pm')
    chem.do_avas()

