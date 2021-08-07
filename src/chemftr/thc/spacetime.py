"""Compute qubit vs toffoli for THC LCU"""
import numpy as np
from numpy.lib.scimath import arccos, arcsin  # want version that has analytic continuation to cplx

from math import pi

from chemftr.util import QR, QI


def qubit_vs_toffoli(lam, dE, eps, n, chi, beta, M, verbose=False):
    """
    Args:
        lam (float) - the lambda-value for the Hamiltonian
        dE (float) - allowable error in phase estimation. usually 0.001
        eps (float) - allowable error for synthesis (dE/(10 * lam)) usually
        n (int) - number of spatial orbitals.
        chi (int) - number of bits of precision for state prep
        beta (int) - number of bits of precision for rotations
        M (int) - THC rank or r_{Thc}
        verbose (bool) - do additional printing of intermediates?

    """
    # (*The number of iterations for the phase estimation.*)
    iters = np.ceil(pi * lam / (dE * 2))
    # (*The number of bits used for each register.*)
    nM = np.ceil(np.log2(M + 1))
    # (*This is the number of distinct items of data we need to output, see Eq. (28).*)
    d = M * (M + 1) / 2 + n / 2
    # (*The number of bits used for the contiguous register.*)
    nc=np.ceil(np.log2(d))
    # (*The output size is 2*Log[M] for the alt values, χ for the keep value, and 2 for the two sign bits.*)
    m=2*nM+2+chi 
    
    # QR[L_,M_]:= np.ceil(MinValue[{L/2^k+M*(2^k-1),k>=0},k∈Integers]] (*This gives the minimum cost for a QROM over L values of size M.*)
    # QRa[L_,M_]:=ArgMin[{L/2^k+M*(2^k-1),k>=0},k∈Integers] (*Gives the optimal k.*)
    # QI[L_]:= np.ceil(MinValue[{L/2^k+2^k,k>=0},k∈Integers]] (*This gives the minimum cost for an inverse QROM over L values.*)
    # QIa[L_]:= np.ceil(ArgMin[{L/2^k+2^k,k>=0},k∈Integers]] (*This gives the minimum cost for an inverse QROM over L values.*)
    
    # (*The next block of code finds the optimal number of bits to use for the rotation angle for the amplitude a
    # mplification taking into account the probability of failure and the cost of the rotations.*)
    oh = np.zeros(20, dtype=float)
    for p in range(1, 20 + 1):
        cos_term = arccos(np.power(2, nM) / np.sqrt(d) / 2)
        # print(cos_term)
        v = np.round(np.power(2, p) / (2 * pi) * cos_term)

        asin_term = arcsin(np.cos(v*2*pi/np.power(2,p)) * np.sqrt(d) / np.power(2, nM))
        sin_term = np.sin(3 * asin_term)**2
        oh[p-1] = (20_000 * (1 / sin_term - 1) + 4 * p).real

    br= np.argmin(oh) + 1  #(*Here br is the number of bits used in the rotation.*)
    # (*Next are the costs for the state preparation.*)
    cp1 = 2 * (10 * nM + 2 * br - 9)
    # (*There is cost 10*Log[M] for preparing the equal superposition over the input registers. This is the costing from above Eq. (29).*)
    cp2 = 2 * (nM ** 2 + nM - 1) # (*This is the cost of computing the contiguous register and inverting it. This is with a sophisticated scheme adding together triplets of bits. This is the cost of step 2 in the list on page 14.*)

    cp3 = QR(d, m)[1] + QI(d)[1] # (*This is the cost of the QROM for the state preparation and its inverse.*)
    cp4 = 2 * chi # (*The cost for the inequality test.*)
    cp5 = 4 * nM # ( *The cost 2*nM for the controlled swaps.*)
    cp6 = 2 * nM + 3  # (*Then there is a cost of nM+1 for swapping the μ and ν registers, where the +3 is because we need to control on two registers, and control swap of the spin registers.*)
    CPCP = cp1 + cp2 + cp3 + cp4 + cp5 + cp6  #  (*The total cost in Eq. (33).*)

    # (*Next are the costs for the select operation.*)
    cs1 = 2 * n  # (*This is the cost of swapping based on the spin register. These costs are from the list on page 15, and this is steps 1 and 7.*)
    k1 = 2 ** QI(M + n / 2)[0]
    cs2a = M + n / 2 - 2 + np.ceil(M / k1) + np.ceil(n / 2 / k1) + k1

    # (*The QROM for the rotation angles the first time.  Here M+n/2-2 is the cost for generating them, and the second part is the cost for inverting them with advanced QROM.*)
    cs2b = M - 2 + QI(M)[1]  # (*The QROM for the rotation angles the second time.  Here the cost M-2 is for generating the angles the second time, and QI[M] is for inverting the QROM. Steps 2 and 6.*)
    cs3 = 4 * n * (beta - 2)  # (*The cost of the rotations  steps 3 and 5.*)
    cs4 = 1  # (*Cost for extra part in making the Z doubly controlled  step 4.*)
    CS = cs1 + cs2a + cs2b + cs3 + cs4  # (*The total select cost in Eq. (43).*)
    costref = 2 * nM + chi + 3  # (*The cost given slightly above Eq. (44) is 2*nM+5. That is a typo and it should have the aleph (equivalent to χ here) like at the top of the column. Here we have +3 in this line, +1 in the next line and +1 for cs4, to give the same total.*)
    cost = CPCP + CS + costref + 1

    # (*Next are qubit costs.*)
    ac1 = 2 * np.ceil(np.log2(iters + 1)) - 1
    ac2 = n
    ac3 = 2 * nM
    ac47 = 5
    ac8 = beta
    ac9 = nc

    kt = 2 ** QR(d, m)[0]
    ac10 = m * kt + np.ceil(np.log2(d / kt))
    ac11 = chi  # (*This is for the equal superposition state to perform the inequality test with the keep register.*)
    ac12 = 1  # (*The qubit to control the swap of the μ and ν registers.*)
    aca = ac1 + ac2 + ac3 + ac47 + ac8 + ac9 + ac11 + ac12
    ac13 = beta * n / 2  # (*This is the data needed for the rotations.*)
    ac14 = beta - 2  # (*These are the ancillas needed for adding into the phase gradient state.*)
    acc = ac13 + ac14 + m  # (*These are the temporary ancillas in between erasing the first QROM ancillas and inverting that QROM. The +m is the for output of the first QROM.*)

    if verbose:
        print("Total Toffoli cost ", cost*iters) # (*The total Toffoli cost.*)
        print("Ancilla for first QROM ", aca+ac10) # (*This is the ancillas needed up to the point we are computing the first QROM.*)
        print("Actual ancilla ... ", np.max([aca+ac10,aca+acc]))  # (*This is the actual ancilla cost if we need more ancillas in between.*)
        print("Spacetime volume ", np.max([aca+ac10,aca+acc])*cost)  # (*Spacetime volume.*)

    # (*First are the numbers of qubits that must be kept throughout the computation. See page 18.*)
    ac1 = 2 * np.ceil(np.log2(iters + 1)) - 1  # (*The qubits used as the control registers for the phase estimation, that must be kept the whole way through. If we used independent controls each time that would increase the Toffoli cost by  np.ceil(np.log2iters+1]]-3, while saving  np.ceil(np.log2iters+1]]-1 qubits.*)
    ac2 = n  # (*The system qubits that must always be included.*)
    ac3 = 2 * nM  # (*The μ and ν registers, that must be kept because they are control registers that aren't fully erased and must be reflected on.*)
    ac4512 = 4  # (*These are the qubits for the spin in the control state as well as the qubit that is rotated for the preparation of the equal superposition state, AND the qubit that is used to control . None of these are fully inversely prepared.*)
    ac8 = beta  # (*The qubits for the phase gradient state.*)
    ac11 = chi  # (*This is for the equal superposition state to perform the inequality test with the keep register. It must be kept around and reflected upon.*)
    perm = ac1 + ac2 + ac3 + ac4512 + ac8 + ac11  # (*The total number of permanent qubits.*)
    # (*In preparing the equal superposition state there are 6 temporary qubits used in the rotation of the ancilla.  There are another three that are needed for the temporary results of inequality tests. By far the largest number, however, come from keeping the temporary ancillas from the inequality tests.  That should be 3*nM+nN-4.  There are an other two qubits in output at the end that will be kept until this step is undone.*)
    nN = np.ceil(np.log2(n / 2))

    qu1 = perm + 4 * nM - 1  # (*This is the maximum number of qubits used while preparing the equal superposition state.*)
    # (*To explain the number of temporary ancillas, we have nM+1 to perform the inequality test on mu and nu with out-of-place addition.  We have another nM-2 for the equality test.  Then we can do the inequality tests on mu and nu with constants (temporarily) overwriting these variables, and keeping nM-1 qubits on each.  Then there are another 2 temporary qubits used for the reflection.  That gives 4*nM-1 total.*)
    tof1 = 10 * nM + 2 * br - 9  # (*This is the number of Toffolis during this step.*)
    perm = perm + 2  # (*This is increasing the running number of permanent ancillas by 2 for the ν=M+1 flag qubit and the success flag qubit.*)

    qu2 = perm + nM**2 + nM  # (*The number of temporary qubits used in this computation is the the same as the number of Toffolis plus one.*)
    tof2 = nM**2 + nM - 1  # (*The Toffoli cost of computing the contiguous register.*)
    perm = perm + nc  # (*The running number of qubits is increased by the number needed for the contiguous register.*)

    kt = 32  # (*Here I'm setting the k-value for the QROM by hand instead of choosing the optimal one for Toffolis.*)
    qu3 = perm + m * kt + np.ceil(np.log2(d / kt))  # (*This is the number of qubits needed during the QROM.*)
    tof3 = np.ceil(d / kt) + m * (kt - 1)  # (*The number of Toffolis for the QROM.*)
    perm = perm + m  # (*The number of ancillas used increases by the actual output size of the QROM.*)

    qu4 = perm + chi  # (*The number of ancilla qubits used for the subtraction for the inequality test.
    # We can use one of the qubits from the registers that are subtracted as the flag qubit so we don't need an extra flag qubit.*)
    tof4 = chi  # (*The number of Toffolis needed for the inequality test. The number of permanent ancillas is unchanged.*)

    qu5 = perm  # (*We don't need any extra ancillas for the controlled swaps.*)
    tof5 = 2 * nM  # (*We are swapping pairs of registers of size nM*)

    qu6 = perm  # (*One extra ancilla for the controlled swap of mu and nu because it is controlled on two qubits.*)
    tof6 = nM + 1  # (*One more Toffoli for the double controls.*)

    qu7 = perm  # (*Swapping based on the spin register.*)
    tof7 = n / 2

    qu8 = perm + nM + beta * n / 2  # (*We use these temporary ancillas for the first QROM for the rotation angles.*)
    tof8 = M + n / 2 - 2  # (*The cost of outputting the rotation angles including those for the one-electron part.*)
    perm = perm + beta * n / 2  # (*We are now need the output rotation angles, though we don't need the temporary qubits from the unary iteration.*)

    qu9 = perm + (beta - 2)  # (*We need a few temporary registers for adding into the phase gradient register.*)
    tof9 = n * (beta - 2)  # (*The cost of the rotations.*)

    qu10 = np.array([-j * beta for j in range(int(n / 2))]) + perm + beta - 2  # Table[-j*beta,{j,0,n/2-1}]+perm+(beta-2) # (*Make a list where we keep subtracting the data qubits that can be erased.*)
    tof10 = np.array([2 * (beta - 2) for j in range(int(n / 2))])  # Table[2*(beta-2),{j,0,n/2-1}]  # (*The cost of the rotations.*)
    perm = perm - beta * n / 2  # (*We've erased the data.*)

    k1 = 2 ** QI(M + n / 2)[0]  # (*Find the k for the phase fixup for the erasure of the rotations.*)
    qu11 = perm + k1 + np.ceil(np.log2(M / k1))  # (*The temporary qubits used. The data qubits were already erased, so don't change perm.*)
    tof11 = np.ceil(M / k1) + np.ceil(n / 2 / k1) + k1

    qu12 = perm  # (*Swapping based on the spin register.*)
    tof12 = n / 2

    qu12a = perm
    tof12a = 1  # (*Swapping the spin registers.*)

    qu13 = perm  # (*Swapping based on the spin register.*)
    tof13 = n / 2

    qu14 = perm + nM - 1 + beta * n / 2  # (*We use these temporary ancillas for the second QROM for the rotation angles.*)
    tof14 = M - 2
    perm = perm + beta * n / 2

    qu15 = perm + (beta - 2)  # (*We need a few temporary registers for adding into the phase gradient register.*)
    tof15 = n * (beta - 2)  # (*The cost of the rotations.*)

    qu16 = perm  # (*Just one Toffoli to do the controlled Z1.*)
    tof16 = 1

    qu17 = np.array([-j * beta for j in range(int(n / 2))]) + perm + beta - 2  # Table[-j*beta,{j,0,n/2-1}]+perm+(beta-2)  # (*Make a list where we keep subtracting the data qubits that can be erased.*)
    tof17 = np.array([2 * (beta - 2) for j in range(int(n / 2))])  # Table[2*(beat-2),{j,0,n/2-1}]  # (*The cost of the rotations.*)
    perm = perm - beta * n / 2  # (*We've erased the data.*)

    k1 = 2 ** QI(M)[0]  # (*Find the k for the phase fixup for the erasure of the rotations.*)
    qu18 = perm + k1 + np.ceil(np.log2(M / k1))  # (*The temporary qubits used. The data qubits were already erased, so don't change perm.*)
    tof18 = np.ceil(M / k1) + k1

    qu19 = perm  # (*Swapping based on the spin register.*)
    tof19 = n / 2

    qu20 = perm + 1  # (*One extra ancilla for the controlled swap of mu and nu because it is controlled on two qubits.*)
    tof20 = nM + 1  # (*One extra Toffoli, because we are controlling on two qubits.*)

    qu21 = perm  # (*We don't need any extra ancillas for the controlled swaps.*)
    tof21 = 2 * nM  # (*We are swapping pairs of registers of size nM*)

    qu22 = perm + chi  # (*The number of ancilla qubits used for the subtraction for the inequality test.
    # We can use one of the qubits from the registers that are subtracted as the flag qubit so we don't need an extra flag qubit.*)
    tof22 = chi  # (*The number of Toffolis needed for inverting the inequality test. The number of permanent ancillas is unchanged.*)
    perm = perm - m  # (*We can erase the data for the QROM for inverting the state preparation, then do the phase fixup.*)

    kt=2**QI(d)[0]
    qu23 = perm + kt + np.ceil(np.log2(d / kt))  # (*This is the number of qubits needed during the QROM.*)
    tof23 = np.ceil(d / kt) + kt  # (*The number of Toffolis for the QROM.*)

    qu24 = perm - nc + nM**2 + nM  # (*The number of temporary qubits used in this computation is the same as the number of Toffolis plus one. We are erasing the contiguous register as we go so can subtract nc.*)
    tof24 = nM**2 + nM - 1  # (*The Toffoli cost of computing the contiguous register.*)
    perm = perm - nc  # (*The contiguous register has now been deleted.*)

    qu25 = perm + 4 * nM - 1  # (*This is the maximum number of qubits used while preparing the equal superposition state.*)
    tof25 = 10 * nM + 2 * br - 9  # (*This is the number of Toffolis during this step.*)
    perm = perm - 2  # (*This is increasing the running number of permanent ancillas by 2 for the ν=M+1 flag qubit and the success flag qubit.*)

    qu26 = perm + costref  # (*We need some ancillas to perform a reflection on multiple qubits. We are including one more Toffoli to make it controlled.*)
    tof26 = costref

    qu27 = perm  # (*Iterate the control register.*)
    tof27 = 1

    qu = np.hstack((np.array([qu1, qu2, qu3, qu4, qu5, qu6, qu7, qu8, qu9]),
                    qu10,
                    np.array([qu11, qu12, qu12a, qu13, qu14, qu15, qu16]),
                    qu17,
                    np.array([qu18, qu19, qu20, qu21, qu22, qu23, qu24, qu25, qu26, qu27])))
    tof = np.hstack((np.array([tof1, tof2, tof3, tof4, tof5, tof6, tof7, tof8, tof9]),
                    tof10,
                    np.array([tof11, tof12, tof12a, tof13, tof14, tof15, tof16]),
                    tof17,
                    np.array([tof18, tof19, tof20, tof21, tof22, tof23, tof24, tof25, tof26, tof27])))
    # cols = np.hstack(([sm, sm, pq, sm, sm, sm, sm, rq, ri]),
    #                    Table[ro, Length[qu10]],
    #                     np.array([rq, sm, sm, sm, rq, ri, sm]),
    #                     Table[ro, Length[qu17]],
    #                  np.array([rq, sm, sm, sm, sm, pq, sm, sm, sm, sm]))

    # NOTE: NOW WE NEED TO PLOT OR GENERATE AN OBJECT TO PLOT. TALK TO DOMINC ABOUT WHAT IS GOING ON IN THE PLOT

if __name__ == "__main__":
    lam = 307.68
    dE = 0.001
    eps = dE / (10 * lam)
    n = 108
    chi = 10
    beta = 16
    M = 350
    qubit_vs_toffoli(lam, dE, eps, n, chi, beta, M, verbose=False)