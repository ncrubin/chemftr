import os
# set mkl thread count for numpy einsum/tensordot calls
# leave one CPU un used  so we can still access this computer
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

import jax.numpy as jnp
# from jax.config import config
# config.update("jax_enable_x64", True)
# from jax.experimental import optimizers
# from  jax import jit, grad

from chemftr.thc.adagrad import adagrad

import h5py
import numpy
import numpy.random
import numpy.linalg

from scipy.optimize import minimize

from uuid import uuid4


def thc_objective_jax(xcur, norb, nthc, eri):
    """
    Loss function for THC factorization using jax numpy

    0.5 \sum_{pqrs}(eri(pqrs) - G(pqrs))^{2}

    G(pqrs) = \sum_{uv}X_{u,p}X_{u,q}Z_{uv}X_{v,r}X_{v,s}

    :param xcur: Current parameters for eta and Z
    :param norb: number of orbitals
    :param nthc: thc-basis dimension
    :param eri: two-electron repulsion integrals in chemist notation
    :return:
    """
    etaPp = xcur[:norb * nthc].reshape(nthc, norb)  # leaf tensor  nthc x norb
    MPQ = xcur[norb * nthc:norb * nthc + nthc * nthc].reshape(nthc, nthc)  # central tensor

    CprP = jnp.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    Iapprox = jnp.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=[(0, 1), (0, 1)])
    deri = eri - Iapprox
    res = 0.5 * jnp.sum((deri) ** 2)
    return res


def thc_objective_grad_jax(xcur, norb, nthc, eri):
    """
    Gradient for THC least-squares objective jax compatible

    :param xcur: Current parameters for eta and Z
    :param norb: number of orbitals
    :param nthc: thc-basis dimension
    :param eri: two-electron repulsion integrals in chemist notation
    :param verbose: optional (False) for print iteration residual and infinity norm
    """
    etaPp = xcur[:norb * nthc].reshape(nthc, norb)  # leaf tensor  nthc x norb
    MPQ = xcur[norb * nthc:norb * nthc + nthc * nthc].reshape(nthc, nthc)  # central tensor

    # m indexes the nthc and p,q,r,s are orbital indices
    CprP = jnp.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    Iapprox = jnp.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=[(0, 1), (0, 1)])
    deri = eri - Iapprox
    res = 0.5 * jnp.sum((deri) ** 2)

    # O(norb^5)
    dL_dZab = -jnp.einsum('pqrs,pqA,rsB->AB', deri, CprP, CprP, optimize=[(0, 1), (0, 1)])
    # O(norb^5)
    dL_dX_GT = -2 * jnp.einsum('Tqrs,Gq,Gv,rsv->GT', deri, etaPp, MPQ, CprP,
                             optimize=[(0, 3), (1, 2), (0, 1)])
    # dL_dX_GT -= jnp.einsum('pTrs,Gp,Gv,rsv->GT', deri, etaPp, MPQ, CprP,
    #                          optimize=[(0, 3), (1, 2), (0, 1)])
    dL_dX_GT -= 2 * jnp.einsum('pqTs,pqu,uG,Gs->GT', deri, CprP, MPQ, etaPp,
                             optimize=[(0, 1), (0, 2), (0, 1)])
    # dL_dX_GT -= jnp.einsum('pqrT,pqu,uG,Gr->GT', deri, CprP, MPQ, etaPp,
    #                          optimize=[(0, 1), (0, 2), (0, 1)])
    return jnp.hstack((dL_dX_GT.ravel(), dL_dZab.ravel()))


def thc_objective(xcur, norb, nthc, eri, verbose=False):
    """
    Loss function for THC factorization

    0.5 \sum_{pqrs}(eri(pqrs) - G(pqrs))^{2}

    G(pqrs) = \sum_{uv}X_{u,p}X_{u,q}Z_{uv}X_{v,r}X_{v,s}

    :param xcur: Current parameters for eta and Z
    :param norb: number of orbitals
    :param nthc: thc-basis dimension
    :param eri: two-electron repulsion integrals in chemist notation
    :param verbose: optional (False) for print iteration residual and infinity norm
    :return:
    """
    etaPp = xcur[:norb*nthc].reshape(nthc,norb)  # leaf tensor  nthc x norb
    MPQ = xcur[norb*nthc:norb*nthc+nthc*nthc].reshape(nthc,nthc) # central tensor

    CprP = numpy.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    Iapprox = numpy.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=['einsum_path', (0, 1), (0, 1)])
    deri = eri - Iapprox
    res = 0.5 * numpy.sum((deri)**2)

    if verbose:
        print("res, max, lambda = {}, {}".format(res, numpy.max(numpy.abs(deri))))

    return res


def thc_objective_grad(xcur, norb, nthc, eri, verbose=False):
    """
    Gradient for THC least-squares objective

    :param xcur: Current parameters for eta and Z
    :param norb: number of orbitals
    :param nthc: thc-basis dimension
    :param eri: two-electron repulsion integrals in chemist notation
    :param verbose: optional (False) for print iteration residual and infinity norm
    """
    etaPp = numpy.array(xcur[:norb*nthc]).reshape(nthc,norb)  # leaf tensor  nthc x norb
    MPQ = numpy.array(xcur[norb*nthc:norb*nthc+nthc*nthc]).reshape(nthc,nthc) # central tensor

    # m indexes the nthc and p,q,r,s are orbital indices
    CprP = numpy.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    Iapprox = numpy.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=['einsum_path', (0, 1), (0, 1)])
    deri = eri - Iapprox
    res = 0.5 * numpy.sum((deri)**2)

    if verbose:
        print("res, max, lambda = {}, {}".format(res, numpy.max(numpy.abs(deri))))

    # O(norb^5)
    dL_dZab = -numpy.einsum('pqrs,pqA,rsB->AB', deri, CprP, CprP, optimize=['einsum_path', (0, 1), (0, 1)])
    # O(norb^5)
    dL_dX_GT = -2 * numpy.einsum('Tqrs,Gq,Gv,rsv->GT', deri, etaPp, MPQ, CprP, optimize=['einsum_path',(0, 3), (1, 2), (0, 1)])
    # dL_dX_GT -= numpy.einsum('pTrs,Gp,Gv,rsv->GT', deri, etaPp, MPQ, CprP, optimize=['einsum_path',(0, 3), (1, 2), (0, 1)])
    dL_dX_GT -= 2 * numpy.einsum('pqTs,pqu,uG,Gs->GT', deri, CprP, MPQ, etaPp, optimize=['einsum_path',(0, 1), (0, 2), (0, 1)])
    # dL_dX_GT -= numpy.einsum('pqrT,pqu,uG,Gr->GT', deri, CprP, MPQ, etaPp, optimize=['einsum_path',(0, 1), (0, 2), (0, 1)])
    return numpy.hstack((dL_dX_GT.ravel(), dL_dZab.ravel()))


def thc_objective_and_grad(xcur, norb, nthc, eri, verbose=False):
    """
    Loss function for THC factorization

    0.5 \sum_{pqrs}(eri(pqrs) - G(pqrs))^{2}

    G(pqrs) = \sum_{uv}X_{u,p}X_{u,q}Z_{uv}X_{v,r}X_{v,s}

    :param xcur: Current parameters for eta and Z
    :param norb: number of orbitals
    :param nthc: thc-basis dimension
    :param eri: two-electron repulsion integrals in chemist notation
    :param verbose: optional (False) for print iteration residual and infinity norm
    :return:
    """
    etaPp = xcur[:norb*nthc].reshape(nthc,norb)  # leaf tensor  nthc x norb
    MPQ = xcur[norb*nthc:norb*nthc+nthc*nthc].reshape(nthc,nthc) # central tensor
    CprP = numpy.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    # path = numpy.einsum_path('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize='optimal')
    Iapprox = numpy.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=['einsum_path', (0, 1), (0, 1)])
    deri = eri - Iapprox
    res = 0.5 * numpy.sum((deri)**2)
    # O(norb^5)
    dL_dZab = -numpy.einsum('pqrs,pqA,rsB->AB', deri, CprP, CprP, optimize=['einsum_path', (0, 1), (0, 1)])
    # O(norb^4 * nthc)
    # leaving the commented out code for documentation purposes
    dL_dX_GT = -2 * numpy.einsum('Tqrs,Gq,Gv,rsv->GT', deri, etaPp, MPQ, CprP, optimize=['einsum_path',(0, 3), (1, 2), (0, 1)])
    # dL_dX_GT -= numpy.einsum('pTrs,Gp,Gv,rsv->GT', deri, etaPp, MPQ, CprP, optimize=['einsum_path',(0, 3), (1, 2), (0, 1)])
    dL_dX_GT -= 2 * numpy.einsum('pqTs,pqu,uG,Gs->GT', deri, CprP, MPQ, etaPp, optimize=['einsum_path',(0, 1), (0, 2), (0, 1)])
    # dL_dX_GT -= numpy.einsum('pqrT,pqu,uG,Gr->GT', deri, CprP, MPQ, etaPp, optimize=['einsum_path',(0, 1), (0, 2), (0, 1)])

    return res, numpy.hstack((dL_dX_GT.ravel(), dL_dZab.ravel()))


class CallBackStore:

    def __init__(self, chkpoint_file, freqency=500):
        """Generic callback function  for storing intermediates from BFGS and Adagrad optimizations"""
        self.chkpoint_file = chkpoint_file
        self.freq = freqency
        self.iter = 0

    def __call__(self, xk):
        if self.iter % self.freq ==  0:
            f = h5py.File(self.chkpoint_file, "w")
            f["xk"] = xk
            f.close()


def lbfgsb_opt_thc(eri, nthc, chkfile_name=None, initial_guess=None, random_seed=None, maxiter=150_000):
    """
    Least-squares fit of two-electron integral tensors with  L-BFGS-B
    """
    # initialize chkfile name if one isn't set
    if chkfile_name is None:
        chkfile_name = str(uuid4()) + '.h5'

    # callback func stores checkpoints
    callback_func = CallBackStore(chkfile_name)

    # set initial guess
    norb = eri.shape[0]
    if initial_guess is None:
        if random_seed is None:
            x = numpy.random.randn(norb*nthc + nthc*nthc)
        else:
            numpy.random.seed(random_seed)
            x = numpy.random.randn(norb*nthc + nthc*nthc)
    else:
        x = initial_guess  # add more checks here for safety

    # L-BFGS-B optimization
    res = minimize(thc_objective_and_grad, x, args=(norb, nthc, eri),  jac=True,
                   method='L-BFGS-B',
                   options={'disp': None, 'maxiter': maxiter,
                            'iprint': 0},
                   callback=callback_func)
    # print(res)
    params = res.x
    x = numpy.array(params)
    f = h5py.File(chkfile_name, "w")
    f["etaPp"] = x[:norb*nthc].reshape(nthc,norb)
    f["ZPQ"] = x[norb*nthc:].reshape(nthc,nthc)
    f.close()
    return params


def adagrad_opt_thc(eri, nthc, chkfile_name=None, initial_guess=None, random_seed=None,
                    stepsize=0.01, momentum=0.9,  maxiter=50_000, gtol=1.0E-5):
    """
    THC opt usually starts with BFGS and then is completed with Adagrad or another
    first order solver.  This  function implements an Adagrad optimization.

    Optimization runs for 50 K iterations.  This is the ONLY stopping cirteria
    used in the FT-THC paper by Lee et al.
    """
    # initialize chkfile name if one isn't set
    if chkfile_name is None:
        chkfile_name = str(uuid4()) + '.h5'

    # callback func stores checkpoints
    callback_func = CallBackStore(chkfile_name)

    # set initial guess
    norb = eri.shape[0]
    if initial_guess is None:
        if random_seed is None:
            x = numpy.random.randn(norb*nthc + nthc*nthc)
        else:
            numpy.random.seed(random_seed)
            x = numpy.random.randn(norb*nthc + nthc*nthc)
    else:
        x = initial_guess  # add more checks here for safety
    opt_init, opt_update, get_params = adagrad(step_size=stepsize, momentum=momentum)
    opt_state = opt_init(x)
    # thc_objective_grad_jit = jit(thc_objective_grad_jax, static_argnums=[1, 2], static_argnames=['norb', 'nthc'])
    # thc_objective_jit = jit(thc_objective_jax, static_argnums=[1, 2], static_argnames=['norb', 'nthc'])
    # ad_obj = grad(thc_objective_jax, argnums=[0])
    # print(ad_obj(x, norb, nthc, eri))
    # print(thc_objective_jit(x, norb, nthc, eri))
    # print(thc_objective_grad_jit(x, norb, nthc, eri))
    # print(type(thc_objective_grad_jit(x, norb, nthc, eri)))

    def update(i, opt_state):
        params = get_params(opt_state)
        gradient = thc_objective_grad(params, norb, nthc, eri)
        grad_norm_l1 = numpy.linalg.norm(gradient,ord=1)
        return opt_update(i, gradient, opt_state), grad_norm_l1

    for t in range(maxiter):
        opt_state, grad_l1 = update(t, opt_state)
        params = get_params(opt_state)
        if t % callback_func.freq == 0:
            # callback_func(params)
            fval = thc_objective(params, norb, nthc,  eri)
            outline = "Objective val {: 5.15f}".format(fval)
            outline += "\tGrad L1-norm {: 5.15f}".format(grad_l1)
            print(outline)
        if grad_l1 <= gtol:
            # break out of loop
            # which sends to save
            break
    else:
        print("Maximum number of iterations reached")
    # save results before returning
    x = numpy.array(params)
    f = h5py.File(chkfile_name, "w")
    f["etaPp"] = x[:norb*nthc].reshape(nthc,norb)
    f["ZPQ"] = x[norb*nthc:].reshape(nthc,nthc)
    f.close()
    return params


def main():
    """This is an example of how to use these functions for THC factor determination"""

    ############################
    #                          #
    #  Define chemical System  #
    #                          #
    ############################

    # #femoco-reiher integrals

    # import chemftr.integrals as chem_ints
    # int_path = chem_ints.__file__.replace('__init__.py', '')
    # hfdump = h5py.File(os.path.join(int_path, 'eri_reiher.h5'), 'r') # './femoco/integrals/eri_reiher.h5','r')
    # eri = hfdump.get('eri')[()]
    # h0 = hfdump.get('h0')[()]
    # hfdump.close()

    # #heme high spin avas hamiltonian integrals
    # fep_files = "/usr/local/google/home/nickrubin/chem/Fe2_porhyrin/FeP_BI/heme_Cys_highspin"
    # hfdump = h5py.File(os.path.join(fep_files, 'avas_hamiltonian_rohf_sto3g_spin5.h5'), 'r')
    # eri = hfdump.get('eri')[()]
    # hfdump.close()

    # #water cluster
    from pyscf import gto, scf, ao2mo
    mol = gto.M()
    mol.atom = [['O', [0.00000,  0.00000,  0.11779]],
                ['H', [0.00000,  0.75545, -0.47116]],
                ['H', [0.00000, -0.75545, -0.47116]]]
    mol.basis = 'sto-3g'
    mol.build()
    mf = scf.RHF(mol)
    mf.run()
    eri = ao2mo.kernel(mol, mf.mo_coeff)
    eri = ao2mo.restore(1, eri, mf.mo_coeff.shape[1])

    norb = eri.shape[0]
    nthc = 3 * norb

    x = numpy.random.randn(norb*nthc + nthc*nthc)

    chkfile = 'water.h5'
    # with h5py.File(chkfile, 'r') as fid:
    #     if 'ZPQ' in list(fid.keys()):
    #         print("loading etaPp and ZPQ")
    #         eta = fid.get('etaPp')[()]
    #         Zpq = fid.get('ZPQ')[()]
    #         x = numpy.hstack((eta.ravel(), Zpq.ravel()))

    #         np_fval, np_grad = thc_objective_and_grad(x, norb, nthc, eri)
    #         # jnp_fval = thc_objective_jax(jnp.array(x), norb, nthc, jnp.array(eri))
    #         # jnp_grad = thc_objective_grad_jax(jnp.array(x), norb, nthc, jnp.array(eri))
    #         # print(np_fval, jnp_fval)
    #         # print(numpy.allclose(np_grad, numpy.array(jnp_grad)))
    #         # print(jnp_grad)
    #         print()
    #     elif 'xk' in list(fid.keys()):
    #         print("loading xk")
    #         x = fid.get('xk')[()]
    #         # np_fval, np_grad = thc_objective_and_grad(x, norb, nthc, eri)
    #         # jnp_fval = thc_objective_jax(jnp.array(x), norb, nthc, jnp.array(eri))
    #         # jnp_grad = thc_objective_grad_jax(jnp.array(x), norb, nthc, jnp.array(eri))

    #         # print(np_fval, jnp_fval)
    #         # print(numpy.allclose(np_grad, numpy.array(jnp_grad)))
    #         # print(jnp_grad)
    #         print()
    x = lbfgsb_opt_thc(eri, nthc, initial_guess=x, chkfile_name=chkfile)
    # I recommend using a diffent checkpoint file for the adagrad part
    chkfile_adagrad = 'water_adagrad.h5'
    # initial step for adagrad will be large. This is fine. We need to escape
    # the L-BFGS hole.
    x = adagrad_opt_thc(eri, nthc, initial_guess=x, chkfile_name=chkfile_adagrad,
                        maxiter=10_000_000, stepsize=0.01)
    exit()



if __name__ == "__main__":
    main()
    # import cProfile
    # cProfile.run('main()', 'einsum_profile.profile')

    # import pstats
    # profile = pstats.Stats('einsum_profile.profile')
    # profile.sort_stats('cumtime')
    # profile.print_stats(30)
