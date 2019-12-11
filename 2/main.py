

from noise import Noise
from filters import Filters
import cv2 as cv

img = './lena.jpg'
noise = Noise(img)


def compare_filter(noise_name, var_list=None, noise_data=None):
    if noise_data is None:
        dist = getattr(noise, noise_name)()
        cv.namedWindow(noise_name, cv.WINDOW_NORMAL)
        cv.imshow(noise_name, dist)
    else:
        dist = noise_data
    if not var_list:
        return
    my_filter = Filters(dist)
    out = None
    for key, value in var_list.items():
        if not value:
            out = my_filter.core(mode=key)
        else:
            out = my_filter.core(mode=key, **value)
        cv.namedWindow(key, cv.WINDOW_NORMAL)
        cv.imshow(key, out)
    cv.waitKey(0)
    return out


def compare_mean_filter():
    mean_vars = {
        'arithmetic_mean': None,
        'geometric_mean': None,
        'harmonic_wave_mean': None,
        'reverse_harmonic_wave_mean': None
    }

    compare_filter("gaussian_noise", var_list=mean_vars)


def reverse_harmonic_wave():
    harmonic_vars = {
        'reverse_harmonic_wave_mean': {
            "q": 1.5
        }
    }
    compare_filter("pepper", var_list=harmonic_vars)

    harmonic_vars2 = {
        'reverse_harmonic_wave_mean': None
    }
    compare_filter("salt", var_list=harmonic_vars2)


def repeat_median():
    dist = noise.salt_and_pepper(salt_vs_pepper=0.1)
    cv.namedWindow('salt_and_pepper', cv.WINDOW_NORMAL)
    cv.imshow('salt_and_pepper', dist)
    my_filter = Filters(dist)
    out = my_filter.core(mode='median_filter')
    cv.namedWindow('1', cv.WINDOW_NORMAL)
    cv.imshow('1', out)
    my_filter1 = Filters(out)
    out1 = my_filter1.core(mode='median_filter')
    cv.namedWindow('2', cv.WINDOW_NORMAL)
    cv.imshow('2', out1)
    my_filter2 = Filters(out1)
    out2 = my_filter2.core(mode='median_filter')
    cv.namedWindow('3', cv.WINDOW_NORMAL)
    cv.imshow('3', out2)
    cv.waitKey(0)


# repeat_median()

def min_and_max():
    harmonic_vars = {
        'max_filter': None
    }
    compare_filter("pepper", var_list=harmonic_vars)

    harmonic_vars2 = {
        'min_filter': None
    }
    compare_filter("salt", var_list=harmonic_vars2)


# min_and_max()


def revision_alpha():
    dist = noise.gaussian_noise()
    cv.namedWindow('gaussian_noise', cv.WINDOW_NORMAL)
    cv.imshow('gaussian_noise', dist)
    add_noise = noise.gaussian_and_salt_and_pepper(salt_vs_pepper=0.1)
    cv.namedWindow('add salt_and_pepper', cv.WINDOW_NORMAL)
    cv.imshow('add salt_and_pepper', add_noise)
    harmonic_vars = {
        'arithmetic_mean': {
            'm': 5,
            'n': 5
        },
        'geometric_mean': {
            'm': 5,
            'n': 5
        },
        'median_filter': {
            'm': 5,
            'n': 5
        },
        'revision_alpha': {
            'm': 5,
            'n': 5,
            'd': 5
        }
    }
    compare_filter("", var_list=harmonic_vars, noise_data=add_noise)


revision_alpha()


def compare_adaptive():
    vars = {
        'median_filter': {
            'm': 7,
            'n': 7
        },
        'adaptive_mean': None
    }
    compare_filter("salt_and_pepper", var_list=vars)
# compare_adaptive()