import numpy as np
import pywt
import scipy.stats


def waveletDenoise(data):
    # data is num_neurons x time_frames

    wavelet = pywt.Wavelet('db4')

    # Determine the maximum number of possible levels for image
    dlen = wavelet.dec_len
    wavelet_levels = pywt.dwt_max_level(data.shape[1], wavelet)

    # Skip coarsest wavelet scales (see Notes in docstring).
    wavelet_levels = max(wavelet_levels - 3, 1)

    data_denoise = np.zeros(np.shape(data))

    shift = 4
    for c in np.arange(-shift, shift + 1):
        data_shift = np.roll(data, c, 1)
        for i in range(np.shape(data)[0]):
            coeffs = pywt.wavedecn(data_shift[i, :], wavelet=wavelet,
                                   level=wavelet_levels)
            # Detail coefficients at each decomposition level
            dcoeffs = coeffs[1:]
            detail_coeffs = dcoeffs[-1]['d']
            # rescaling using a single estimation of level noise based on first level coefficients.
            # Consider regions with detail coefficients exactly zero to be masked out
            # detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]
            # 75th quantile of the underlying, symmetric noise distribution
            denom = scipy.stats.norm.ppf(0.75)
            sigma = np.median(np.abs(detail_coeffs)) / denom
            np.shape(sigma)
            sigma_mat = np.tile(sigma, (wavelet_levels, 1))
            np.shape(sigma_mat)

            tot_num_coeffs = pywt.wavedecn_size(coeffs)
            # universal threshold
            threshold = np.sqrt(2 * np.log(tot_num_coeffs))
            threshold = sigma * threshold

            denoised_detail = [{key: pywt.threshold(level[key],
                                                    value=threshold,
                                                    mode='hard') for key in level}
                               for level in dcoeffs]

            # Dict of unique threshold coefficients for each detail coeff. array

            denoised_coeffs = [coeffs[0]] + denoised_detail

            data_denoise[i, :] = data_denoise[i, :] + np.roll(
                pywt.waverecn(denoised_coeffs, wavelet), -c)[:data_denoise.shape[1]]

    data_denoise = data_denoise / (2 * shift + 1)
    return data_denoise
