import sys
import h5py
import numpy as np
from chemftr import sf, df
from chemftr.rank_reduce import thc_via_cp3
from chemftr.util import ccsd_t, read_cas
import chemftr.integrals as chemftr_integrals
from chemftr.thc.thc_factorization import lbfgsb_opt_thc, adagrad_opt_thc, lbfgsb_opt_thc_l2reg
from chemftr.thc.computing_lambda_thc import compute_thc_lambda
from chemftr.thc.costing_thc import cost_thc


def main():
    INTS = 'hamiltonian_heme_cys_mult6_rohf_ccpvdz_spin5_pm.h5'
    name = 'heme_cys_mult6_rohf_ccpvdz_spin5'
    filename = 'thc_factorization_' + name + '.txt'
    USE_KERNEL = True
    RUN_LBFGSB = True
    RUN_REG = True
    thc_ranks = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    with h5py.File(INTS, "r") as f:
        num_alpha = int(f['active_nalpha'][()])
        num_beta = int(f['active_nbeta'][()])
        try:
            num_orb = np.asarray(f['h0'][()]).shape[0]
        except KeyError:
            num_orb = np.asarray(f['h1'][()]).shape[0]

    cas_info = "CAS((%sa, %sb), %so)" % (num_alpha, num_beta, num_orb)

    h1, eri_full, ecore, num_alpha, num_beta = read_cas(INTS, num_alpha, num_beta)

    # Reference calculation (dim = None is full cholesky / exact ERIs)
    # run silently
    # with RunSilent():
    escf, ecor, etot = sf.compute_ccsd_t(cholesky_dim=None, integral_path=INTS, use_kernel=USE_KERNEL)
    exact_ecor = ecor
    exact_etot = etot

    with open(filename, 'w') as f:
        print("\n THC factorization data for '" + name + "'.", file=f)
        print("    [*] using " + cas_info, file=f)
        print("        [+]                      E(SCF): %18.8f" % escf, file=f)
        print("        [+] Active space CCSD(T) E(cor): %18.8f" % ecor, file=f)
        print("        [+] Active space CCSD(T) E(tot): %18.8f" % etot, file=f)
        print("{}".format('=' * 120), file=f)
        print(
            "{:^12} {:^12} {:^12} {:^24} {:^20} {:^20}".format('nthc', 'lambda', '||eri-Delta||', 'CCSD(T) error (mEh)',
                                                               'logical qubits', 'Toffoli count'), file=f)
        print("{}".format('-' * 120), file=f)

    for nthc in thc_ranks:
        eri_cp3, beta, gamma = thc_via_cp3(eri_full=eri_full, nthc=nthc, first_factor_thresh=1.0E-8,
                                           perform_bfgs_opt=False,
                                           conv_eps=1.0E-5, random_start_thc=False)
        # Tell me how different the eri's are after CP3 optimization
        print("L2-norm cp3 ", np.linalg.norm(eri_cp3 - eri_full) ** 2)

        # set variables for next run
        x = np.hstack((beta.ravel(), gamma.ravel()))
        etaPp = x[:num_orb * nthc].reshape(nthc, num_orb)
        MPQ = x[num_orb * nthc:].reshape(nthc, nthc)
        CprP = np.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
        eri_thc = np.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=['einsum_path', (0, 1), (0, 1)])

        if RUN_LBFGSB:
            chkfile = 'temp.h5'
            if RUN_REG:
                x = lbfgsb_opt_thc_l2reg(eri_full, nthc, initial_guess=x, chkfile_name=chkfile, maxiter=10_000,
                                         disp=True, penalty_param=None)
            else:
                x = lbfgsb_opt_thc(eri_full, nthc, initial_guess=x, chkfile_name=chkfile, maxiter=5_000, disp=True)

            sys.stdout.flush()

            # # I recommend using a diffent checkpoint file for the adagrad part
            # chkfile_adagrad = 'temp_adagrad.h5'
            # # initial step for adagrad will be large. This is fine. We need to escape
            # # the L-BFGS hole.
            # x = np.array(x, dtype=np.single)
            # eri = np.array(eri_full, dtype=np.single)
            # x = adagrad_opt_thc(eri, nthc, initial_guess=x, chkfile_name=chkfile_adagrad,
            #                     maxiter=5_000, stepsize=0.01)

            # reset the eri_thc variable
            x = np.array(x)
            etaPp = x[:num_orb * nthc].reshape(nthc, num_orb)
            MPQ = x[num_orb * nthc:].reshape(nthc, nthc)
            CprP = np.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
            # path = numpy.einsum_path('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize='optimal')
            eri_thc = np.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=['einsum_path', (0, 1), (0, 1)])

            # compute L2 norm difference
            print("L2-norm LBFGSB ", np.linalg.norm(eri_thc - eri_full) ** 2)

        thc_save_file = "THC_tensor_{}_cp3".format(nthc)
        if RUN_LBFGSB:
            thc_save_file += "_run_lbfgs.h5"
        else:
            thc_save_file += ".h5"
        with h5py.File(thc_save_file, 'w') as fid:
            fid.create_dataset('etaPp', data=etaPp)
            fid.create_dataset('MPQ', data=MPQ)
            fid.create_dataset('eri_thc', data=eri_thc)

        approx_escf, approx_ecor, approx_etot = ccsd_t(h1, eri_thc, ecore, num_alpha, num_beta, eri_full=eri_full,
                                                       use_kernel=USE_KERNEL)
        error = (approx_etot - exact_etot) * 1E3  # to mEh
        l2_norm_error_eri = np.linalg.norm(eri_thc - eri_full)

        _, _, _, lambda_T, lambda_z, lambda_tot = compute_thc_lambda(h1, etaPp, MPQ, eri_full,
                                                                     use_eri_reconstruct_for_v=False)
        DE = 0.001
        CHI = 10
        N = num_orb
        LAM = lambda_tot
        BETA = 20
        THC_DIM = nthc
        # Here we're using an initial calculation with a very rough estimate of the number of steps
        # to give a more accurate number of steps. Then we input that into the function again.
        output = cost_thc(N, LAM, DE, CHI, BETA, THC_DIM, stps=20000)
        stps2 = output[0]
        output = cost_thc(N, LAM, DE, CHI, BETA, THC_DIM, stps2)
        toffoli_count, logical_qubits = output[1], output[2]

        with open(filename, 'a') as f:
            print("{:^12} {:^12.1f} {:^12.8f} {:^24.2f} {:^20} {:^20.1e}".format(nthc, LAM, l2_norm_error_eri, error,
                                                                                 logical_qubits,
                                                                                 toffoli_count), file=f)


main()