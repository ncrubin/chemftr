import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)
from jax.config import config
config.update("jax_enable_x64", True)
from jax.experimental import optimizers

import h5py
import numpy
import numpy.random
import numpy.linalg

import jax
# import jax.numpy as np
import numpy as np


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

    CprP = np.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    Iapprox = np.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=['einsum_path', (0, 1), (0, 1)])
    deri = eri - Iapprox
    res = 0.5 * np.sum((deri)**2)

    if verbose:
        print("res, max, lambda = {}, {}".format(res, np.max(np.abs(deri))))# .aval.val))

    return res


def thc_objective_grad(xcur, norb, nthc, eri, verbose=False):
    etaPp = np.array(xcur[:norb*nthc]).reshape(nthc,norb)  # leaf tensor  nthc x norb
    MPQ = np.array(xcur[norb*nthc:norb*nthc+nthc*nthc]).reshape(nthc,nthc) # central tensor

    # m indexes the nthc and p,q,r,s are orbital indices
    CprP = np.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    Iapprox = np.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=['einsum_path', (0, 1), (0, 1)])
    deri = eri - Iapprox
    res = 0.5 * np.sum((deri)**2)

    if verbose:
        print("res, max, lambda = {}, {}".format(res, np.max(np.abs(deri))))

    # O(norb^5)
    dL_dZab = -np.einsum('pqrs,pqA,rsB->AB', deri, CprP, CprP, optimize=['einsum_path', (0, 1), (0, 1)])
    # O(norb^5)
    dL_dX_GT = -np.einsum('Tqrs,Gq,Gv,rsv->GT', deri, etaPp, MPQ, CprP, optimize=['einsum_path',(0, 3), (1, 2), (0, 1)])
    dL_dX_GT -= np.einsum('pTrs,Gp,Gv,rsv->GT', deri, etaPp, MPQ, CprP, optimize=['einsum_path',(0, 3), (1, 2), (0, 1)])
    dL_dX_GT -= np.einsum('pqTs,pqu,uG,Gs->GT', deri, CprP, MPQ, etaPp, optimize=['einsum_path',(0, 1), (0, 2), (0, 1)])
    dL_dX_GT -= np.einsum('pqrT,pqu,uG,Gr->GT', deri, CprP, MPQ, etaPp, optimize=['einsum_path',(0, 1), (0, 2), (0, 1)])
    return np.hstack((dL_dX_GT.ravel(), dL_dZab.ravel()))


def callback(x, t):
    f = h5py.File("thc_coupled_bfgs_chk.h5", "w")
    f["x"] = x
    f.close()


def main():
    hfdump = h5py.File('./femoco/integrals/eri_reiher.h5','r')
    eri = hfdump.get('eri')[()]
    h0 = hfdump.get('h0')[()]
    hfdump.close()

    norb = eri.shape[0]

    nthc = 250

    numpy.random.seed(0)
    x = numpy.random.randn(norb*nthc + nthc*nthc)#  + norb**2)
    x_jax = np.array(x)

    opt_init, opt_update, get_params = optimizers.adagrad(step_size=5.0)
    opt_state = opt_init(x_jax)

    # # obj_grad = jax.grad(thc_objective, argnums=0)

    # def update(i, opt_state):
    #     params = get_params(opt_state)
    #     # gradient = jax.grad(objective_function)(params, norb, nthc, eri)
    #     # gradient = obj_grad(params, norb, nthc, eri)
    #     # test_gradient = thc_objective_grad(params, norb, nthc, eri)
    #     # assert np.allclose(gradient, test_gradient)
    #     gradient = thc_objective_grad(params, norb, nthc, eri)
    #     return opt_update(i, gradient, opt_state)

    # for t in range(50_000):
    #     print("{}    ".format(t),  end='')
    #     opt_state = update(t, opt_state)
    #     params = get_params(opt_state)
    #     callback(params, t)

    from scipy.optimize import minimize

    res = minimize(thc_objective, x, args=(norb, nthc, eri),  jac=thc_objective_grad, method='L-BFGS-B', options={'disp': True})
    params = res.x

    x = numpy.array(params)
    f = h5py.File("thc_coupled_bfgs_optimized.h5", "w")
    f["etaPp"] = x[:norb*nthc].reshape(nthc,norb)
    f["ZPQ"] = x[norb*nthc:].reshape(nthc,nthc)
    # f["U"] = x[norb*nthc+nthc**2:].reshape(norb,norb)
    f.close()


if __name__ == "__main__":
    
    
    main()
    # import cProfile
    # cProfile.run('main()', 'einsum_profile.profile')

    # import pstats
    # profile = pstats.Stats('einsum_profile.profile')
    # profile.sort_stats('cumtime')
    # profile.print_stats(30)
