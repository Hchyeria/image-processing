# Noise Package
# Author: Hchyeria

import math
import numpy as np
import sys
sys.setrecursionlimit(10000)


class Filters:
    def __init__(self, img):
        self.img = img

    def core(self, mode=None, **kwargs):
        allowedtypes = {
            'arithmetic_mean': 0,
            'geometric_mean': 1,
            'harmonic_wave_mean': 2,
            'reverse_harmonic_wave_mean': 3,
            'median_filter': 4,
            'max_filter': 5,
            'min_filter': 6,
            'middle_filter': 7,
            'revision_alpha': 8,
            'adaptive_mean': 9}

        kwdefaults = {
            'm': 2,
            'n': 2,
            'q': -1.5,
            'd': 2,
            'max_size': 7,
            'step': 2}

        allowedkwargs = {
            0: ['m', 'n'],
            1: ['m', 'n'],
            2: ['m', 'n'],
            3: ['m', 'n', 'q'],
            4: ['m', 'n'],
            5: ['m', 'n'],
            6: ['m', 'n'],
            7: ['m', 'n'],
            8: ['m', 'n', 'd'],
            9: ['m', 'n', 'max_size', 'step']}

        for key in kwargs:
            if key not in allowedkwargs[allowedtypes[mode]]:
                raise ValueError('%s keyword not in allowed keywords %s' %
                                 (key, allowedkwargs[allowedtypes[mode]]))

        # Set kwarg defaults
        for kw in allowedkwargs[allowedtypes[mode]]:
            kwargs.setdefault(kw, kwdefaults[kw])

        image = self.img.copy()
        h, w, c = image.shape
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    m, n = kwargs['m'], kwargs['n']
                    temp = image[i:i + m, j:j + n, k]
                    if allowedtypes[mode] == 0:
                        image[i][j][k] = np.mean(temp)
                    elif allowedtypes[mode] == 1:
                        image[i][j][k] = np.prod(np.power(temp, 1 / (m * n)))
                    elif allowedtypes[mode] == 2:
                        image[i][j][k] = (m * n) / (np.sum(1 / temp))
                    elif allowedtypes[mode] == 3:

                        image[i][j][k] = np.sum(temp ** (kwargs['q'] + 1)) / np.sum(temp ** kwargs['q'])
                    elif allowedtypes[mode] == 4:
                        image[i][j][k] = np.median(temp)
                    elif allowedtypes[mode] == 5:
                        image[i][j][k] = np.max(temp)
                    elif allowedtypes[mode] == 6:
                        image[i][j][k] = np.min(temp)
                    elif allowedtypes[mode] == 7:
                        image[i][j][k] = (np.min(temp) + np.max(temp)) / 2
                    elif allowedtypes[mode] == 8:
                        if kwargs['d'] < 0 or kwargs['d'] > m * n - 1:
                            raise ValueError('%d < 0 or > m * n - 1',
                                             kwargs['d'])
                        temp = np.sort(temp.reshape(temp.shape[0] * temp.shape[1], 1), axis=0)
                        min_num = math.ceil(kwargs['d'] / 2)
                        max_num = kwargs['d'] // 2
                        temp = temp[min_num: temp.shape[0]*temp.shape[1] - max_num]
                        image[i][j][k] = np.mean(temp)
                    elif allowedtypes[mode] == 9:
                        image[i][j][k] = adaptive_mean(image, i, j, k, m=kwargs['m'], n=kwargs['n'], max_size=kwargs['max_size'])

        return image


def adaptive_mean(image, i, j, k, min_size=3, m=None, n=None, max_size=7, step=2):
    if m is not None and n is not None:
        min_size = m * n
    m = math.ceil(min_size / 2)
    n = min_size // 2
    zxy = image[i][j][k]
    temp = image[i:i + m, j:j + n, k]
    kernel_size = temp.shape[0] * temp.shape[1]
    temp = np.sort(temp.reshape(kernel_size, 1), axis=0)
    min_val = temp[0]
    max_val = temp[kernel_size - 1]
    med_val = temp[kernel_size // 2]
    if min_val < med_val < max_val:
        # judge B
        if min_val < zxy < max_val:
            return zxy
        else:
            return med_val
    else:
        # enlarge the kernel size
        min_size += step
        if min_size <= max_size:
            return adaptive_mean(image, i, j, k, min_size=min_size, max_size=max_size)
        else:
            return med_val
