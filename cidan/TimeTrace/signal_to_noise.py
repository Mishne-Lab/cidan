from itertools import groupby

import numpy as np


def trial_Pnoise(X, **kwargs):
    W = kwargs['W'] if 'W' in kwargs else 25
    step = kwargs['step'] if 'step' in kwargs else 10
    N = X.shape[1]

    # calculate P_Noise
    #     P_Noise = utils.make2dList(X.shape[0], 2)
    #     P_Noise = [None] * X.shape[0]
    P_Noise = np.zeros((X.shape[0], 2))
    for trial in range(X.shape[0]):
        record_std = []
        record_mean = []
        for w in range((N - W) // step + 1):
            window = w * step + np.linspace(0, W, W, endpoint=False, dtype=int)
            #             record.append(np.std(X[trial,window], ddof=1))
            record_std.append(np.std(X[trial, window]))
            record_mean.append(np.mean(X[trial, window]))
        medIdx = record_std.index(
            np.percentile(record_std, 25, interpolation='nearest'))
        P_Noise[trial, [0, 1]] = [record_mean[medIdx], record_std[medIdx]]
    #         P_Noise[trial] = np.median(record)

    return P_Noise


def trial_PSNR(X, **kwargs):
    Peak_Sig = np.amax(X, axis=1, keepdims=True)
    #     P_Sig = np.linalg.norm(X_Str[0], keepdims=True, axis=1)**2 / N
    if 'Pnoise' in kwargs:
        P_Noise = kwargs['Pnoise']
    else:
        W = kwargs['W'] if 'W' in kwargs else 25
        step = kwargs['step'] if 'step' in kwargs else 10
        P_Noise = trial_Pnoise(X, W=W, step=step)[:, 1]

    #     SNR = np.log10(Peak_Sig/P_Noise)
    SNR = Peak_Sig / P_Noise
    return SNR.flatten()


def trial_active(X, **kwargs):
    activeTrial_list = []
    # nSTD - # of std above mean for activity threshold
    # width - # of time frames for activity threshold
    nSTD = kwargs['n'] if 'n' in kwargs else 1
    width = kwargs['w'] if 'w' in kwargs else 4

    noise_stats = trial_Pnoise(X, W=25, step=10)

    for trialId in range(X.shape[0]):
        trial = X[trialId, :]
        meanNoise, stdNoise = noise_stats[trialId, :]
        T = trial > (meanNoise + nSTD * stdNoise)
        for k, g in groupby(T):
            if k:
                length = sum(1 for _ in g)
                if length >= width:
                    activeTrial_list.append(trialId)
                    break
    #     temp = [noise_stats[i] for i in activeTrial_list]
    #     return X[activeTrial_list,:], noise_stats[activeTrial_list,:]
    return noise_stats, activeTrial_list
