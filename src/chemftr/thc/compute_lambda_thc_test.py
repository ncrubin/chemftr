import chemftr.integrals as int_folder
from chemftr import thc
from chemftr.molecule import load_casfile_to_pyscf

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

    mol, mf = load_casfile_to_pyscf(eri_file, num_alpha = 27, num_beta = 27)

    lambda_tot, nthc, sqrt_res, res, lambda_T, lambda_z  = \
        thc.compute_lambda(mf, etaPp=etaPp, MPQ=MPQ)
    assert nthc == 250
    assert np.isclose(np.round(lambda_tot), 294)

test_lambda()
