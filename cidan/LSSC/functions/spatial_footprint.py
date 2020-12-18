"""
From Calman: https://github.com/flatironinstitute/CaImAn
Licensed under gpl v2 license
"""

import logging
from typing import Any, List, Tuple

import numpy as np
import scipy
from past.utils import old_div
from scipy.sparse import csc_matrix
from scipy.stats import norm


# %%
def find_activity_intervals(C, Npeaks: int = 5, tB=-3, tA=10,
                            thres: float = 0.3) -> List:
    # todo todocument
    import peakutils
    K, T = np.shape(C)
    L: List = []
    for i in range(K):
        if np.sum(np.abs(np.diff(C[i, :]))) == 0:
            L.append([])
            logging.debug('empty component at:' + str(i))
            continue
        indexes = peakutils.indexes(C[i, :], thres=thres)
        srt_ind = indexes[np.argsort(C[i, indexes])][::-1]
        srt_ind = srt_ind[:Npeaks]
        L.append(srt_ind)

    LOC = []
    for i in range(K):
        if len(L[i]) > 0:
            interval = np.kron(L[i], np.ones(int(np.round(tA - tB)), dtype=int)) + \
                       np.kron(np.ones(len(L[i]), dtype=int), np.arange(tB, tA))
            interval[interval < 0] = 0
            interval[interval > T - 1] = T - 1
            LOC.append(np.array(list(set(interval))))
        else:
            LOC.append(None)

    return LOC


# %%
def classify_components_ep(Y, A, C, Athresh=0.1, Npeaks=5, tB=-3, tA=10, thres=0.3) -> \
Tuple[np.ndarray, List]:
    """Computes the space correlation values between the detected spatial
    footprints and the original data when background and neighboring component
    activity has been removed.
    Args:
        Y: list [ndarray]
            each trial x,y,t

        A: scipy sparse array
            spatial components

        C: ndarray
            Fluorescence traces

        Athresh: float
            Degree of spatial overlap for a neighboring component to be
            considered overlapping

        Npeaks: int
            Number of peaks to consider for computing spatial correlation

        tB: int
            Number of frames to include before peak

        tA: int
            Number of frames to include after peak

        thres: float
            threshold value for computing distinct peaks

    Returns:
        rval: ndarray
            Space correlation values

        significant_samples: list
            Frames that were used for computing correlation values
    """
    Y = [x.reshape((x.shape[0], -1), order="C").transpose() for x in Y]
    K, _ = np.shape(C)
    A = csc_matrix(A)
    AA = (A.T * A).toarray()
    nA = np.sqrt(np.array(A.power(2).sum(0)))
    AA = old_div(AA, np.outer(nA, nA.T))
    AA -= np.eye(K)

    LOC = find_activity_intervals(C, Npeaks=Npeaks, tB=tB, tA=tA, thres=thres)
    rval = np.zeros(K)

    significant_samples: List[Any] = []
    for i in range(K):
        if (i + 1) % 200 == 0:  # Show status periodically
            logging.info('Components evaluated:' + str(i))
        if LOC[i] is not None:
            atemp = A[:, i].toarray().flatten()
            atemp[np.isnan(atemp)] = np.nanmean(atemp)
            ovlp_cmp = np.where(AA[:, i] > Athresh)[0]
            indexes = set(LOC[i])
            for _, j in enumerate(ovlp_cmp):
                if LOC[j] is not None:
                    indexes = indexes - set(LOC[j])

            if len(indexes) == 0:
                indexes = set(LOC[i])
                logging.warning('Component {0} is only active '.format(i) +
                                'jointly with neighboring components. Space ' +
                                'correlation calculation might be unreliable.')

            indexes = np.array(list(indexes)).astype(np.int)
            px = np.where(atemp > 0)[0]
            if px.size < 3:
                logging.warning('Component {0} is almost empty. '.format(
                    i) + 'Space correlation is set to 0.')
                rval[i] = 0
                significant_samples.append({0})
            else:
                ysqr = np.array(np.hstack([x[px, :] for x in Y]))
                ysqr[np.isnan(ysqr)] = np.nanmean(ysqr)
                mY = np.mean(ysqr[:, indexes], axis=-1)
                significant_samples.append(indexes)
                rval[i] = scipy.stats.pearsonr(mY, atemp[px])[0]

        else:
            rval[i] = 0
            significant_samples.append(0)

    return rval, significant_samples
