"""To finish"""
for M in range(250, 820, 50):
    f = h5py.File("THC_tensors_approximated/reiher/M_%i_beta_16_eta_10.h5" % M, "r")
    MPQ = f["MPQ"][()]  # nthc x nthc
    etaPp = f["etaPp"][()]
    f.close()

    CprP = numpy.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    BprQ = numpy.tensordot(CprP, MPQ, axes=([2], [0]))
    Iapprox = numpy.tensordot(CprP, numpy.transpose(BprQ), axes=([2], [0]))
    deri = eri - Iapprox
    res = 0.5 * numpy.sum((deri) ** 2)

    eri_thc = numpy.einsum("Pp,Pr,Qq,Qs,PQ->prqs", etaPp, etaPp, etaPp, etaPp, MPQ, optimize=True)

    SPQ = etaPp.dot(etaPp.T)
    cP = numpy.diag(numpy.diag(SPQ))
    MPQ_normalized = cP.dot(MPQ).dot(cP)

    lambda_z = numpy.sum(numpy.abs(MPQ_normalized)) * 0.5
    T = h0 - 0.5 * numpy.einsum("illj->ij", eri) + numpy.einsum("llij->ij", eri_thc)
    e, v = numpy.linalg.eigh(T)
    lambda_T = numpy.sum(numpy.abs(e))

    lambda_tot = lambda_z + lambda_T

    print(M, numpy.sqrt(res), res, lambda_T, lambda_z, lambda_tot)