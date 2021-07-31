import chemftr.integrals as int_folder
from chemftr.thc.computing_lambda_thc import compute_thc_lambda

import os

import h5py

import numpy as np


def test_lambda():
    integral_path = int_folder.__file__.replace('__init__.py', '')
    thc_factor_file = os.path.join(integral_path, 'M_250_beta_16_eta_10.h5')
    eri_file = os.path.join(integral_path,'eri_reiher.h5')
    with h5py.File(thc_factor_file, 'r') as fid:
        MPQ = fid['MPQ'][...]
        etaPp = fid['etaPp'][...]

    with h5py.File(eri_file, 'r') as fid:
        eri = fid['eri'][...]
        h0 = fid['h0'][...]

    nthc, sqrt_res, res, lambda_T, lambda_z, lambda_tot = \
        compute_thc_lambda(oei=h0, etaPp=etaPp, MPQ=MPQ, true_eri=eri, use_eri_reconstruct_for_v=False)
    assert nthc == 250
    assert np.isclose(np.round(lambda_tot), 294)