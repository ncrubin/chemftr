import os
# set mkl thread count for numpy einsum/tensordot calls
# leave one CPU un used  so we can still access this computer
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

from chemftr.thc.adagrad import adagrad
from chemftr.thc.thc_objectives import (thc_objective, thc_objective_grad, thc_objective_and_grad,
                                        cp_ls_cholesky_factor_objective, )

import h5py
import numpy
import numpy.random
import numpy.linalg

from scipy.optimize import minimize

from uuid import uuid4



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


def lbfgsb_opt_thc(eri, nthc, chkfile_name=None, initial_guess=None, random_seed=None, maxiter=150_000,
                   disp=False):
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
                   options={'disp': disp, 'maxiter': maxiter},
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


def lbfgsb_opt_cholesky(cholesky_factor, nthc, chkfile_name=None, initial_guess=None, random_seed=None):
    """
    Least-squares fit of cholesky tensors with  L-BFGS-B

    cholesky_factor is reshaped into (norb, norb, num_cholesky)
    """
    # initialize chkfile name if one isn't set
    if chkfile_name is None:
        chkfile_name = str(uuid4()) + '.h5'

    # callback func stores checkpoints
    callback_func = CallBackStore(chkfile_name)

    # set initial guess
    norb = cholesky_factor.shape[0]
    ncholfactor = cholesky_factor.shape[-1]

    if initial_guess is None:
        if random_seed is None:
            x = numpy.random.randn(norb*nthc + nthc*ncholfactor)
        else:
            numpy.random.seed(random_seed)
            x = numpy.random.randn(norb*nthc + nthc*ncholfactor)
    else:
        x = initial_guess  # add more checks here for safety

    # L-BFGS-B optimization
    res = minimize(cp_ls_cholesky_factor_objective, x, args=(norb, nthc, cholesky_factor, True),  jac=True,
                   method='L-BFGS-B', options={'disp': True, 'ftol': 1.0E-4}, callback=callback_func)
    print(res)
    params = res.x
    x = numpy.array(params)
    f = h5py.File(chkfile_name, "w")
    f["etaPp"] = x[:norb*nthc].reshape(nthc,norb)
    f["gamma"] = x[norb*nthc:].reshape(nthc,cholesky_factor.shape[-1])
    f.close()
    return params


