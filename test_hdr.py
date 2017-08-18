from hdr import hdr_boxplot
import pytest
import numpy as np
import numpy.testing as npt
from mock import patch

import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')


# Water surface temperature data from:
# https://www.math.univ-toulouse.fr/~ferraty/SOFTWARES/NPFDA/npfda-datasets.html
# http://www.cpc.ncep.noaa.gov/data/indices/
# data = np.loadtxt('data/elnino.dat')
data = sm.datasets.elnino.load()
labels = data.raw_data[:, 0]
data = data.raw_data[:, 1:]
# print('Data shape: ', data.shape)


def test_basic():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    output = hdr_boxplot(data, labels=labels, ax=ax, xdata=np.linspace(1, 12, 12))
    fig, median, outliers, extreme_quartile, mean_quartile, extra_quartiles = output
    assert extra_quartiles is None

    ax.set_xlabel("Month of the year")
    ax.set_ylabel("Sea surface temperature (C)")
    ax.set_xticks(np.arange(13, step=3) - 1)
    ax.set_xticklabels(["", "Mar", "Jun", "Sep", "Dec"])
    ax.set_xlim([-0.2, 11.2])
    plt.show()

    median_t = [22.32,  21.17,  20.26,  20.,  20.37,  21.13,  22.28,  24.1,
                25.67,  26.18,  25.26,  23.96]
    npt.assert_almost_equal(median, median_t, decimal=2)

    quarts = np.vstack([outliers, extreme_quartile, mean_quartile])
    quarts_t = np.vstack([[[23.740, 23.100, 22.223, 21.972, 22.595, 23.560,
                            25.020, 26.720, 28.196, 29.124, 29.070, 28.397],
                           [25.928, 24.633, 23.177, 22.480, 22.678, 23.325,
                            24.242, 25.189, 25.958, 25.783, 23.995, 22.598],
                           [26.611, 25.935, 24.658, 24.082, 24.609, 25.513,
                            26.849, 27.883, 28.749, 29.222, 28.644, 27.979]], 
                          [[25.131, 23.918, 22.605, 22.020, 22.285, 22.980,
                            24.032, 25.342, 26.896, 27.796, 27.641, 26.692], 
                           [20.397, 19.356, 18.752, 18.738, 19.223, 20.067,
                            21.185, 23.013, 24.565, 24.762, 23.123, 21.526]],
                          [[23.582, 22.364, 21.280, 20.879, 21.206, 21.941,
                            23.060, 24.734, 26.305, 27.053, 26.549, 25.441],
                           [21.239, 20.145, 19.382, 19.243, 19.653, 20.444,
                            21.572, 23.462, 25.057, 25.377, 24.066, 22.597]]])

    npt.assert_almost_equal(quarts, quarts_t, decimal=2)


@patch("matplotlib.pyplot.show")
def test_alpha(mock_show):
    output = hdr_boxplot(data, alpha=[0.7])
    extra_quarts = output[-1]
    extra_quarts_t = np.vstack([[[24.13, 22.9 , 21.73, 21.27, 21.57, 22.29,
                                  23.38, 24.95, 26.53, 27.35, 26.95, 25.9 ],
                                 [20.89, 19.83, 19.11, 19.01, 19.42, 20.22,
                                  21.34, 23.27, 24.85, 25.13, 23.72, 22.2 ]]])
    npt.assert_almost_equal(extra_quarts, extra_quarts_t, decimal=2)


@patch("matplotlib.pyplot.show")
def test_multiple_alpha(mock_show):
    output = hdr_boxplot(data, alpha=[0.8, 0.6])
    extra_quarts = output[-1]
    extra_quarts_t = [[24.496, 23.251, 22.032, 21.524, 21.811, 22.521,
                       23.589, 25.064, 26.650, 27.488, 27.195, 26.180],
                      [20.707, 19.655, 18.961, 18.876, 19.301, 20.101,
                       21.223, 23.156, 24.748, 25.006, 23.519, 21.972],
                      [23.765, 22.542, 21.430, 21.008, 21.327, 22.057,
                       23.166, 24.796, 26.386, 27.127, 26.698, 25.607],
                      [21.112, 20.010, 19.262, 19.134, 19.544, 20.333,
                       21.460, 23.376, 24.994, 25.303, 23.917, 22.431]]

    npt.assert_almost_equal(extra_quarts, np.vstack(extra_quarts_t), decimal=2)


@patch("matplotlib.pyplot.show")
def test_threshold(mock_show):
    output = hdr_boxplot(data, alpha=[0.8], threshold=0.97)
    outliers = output[1]
    outliers_t = np.vstack([[23.740, 23.100, 22.223, 21.972, 22.595, 23.560,
                             25.020, 26.720, 28.196, 29.124, 29.070, 28.397],
                            [26.611, 25.935, 24.658, 24.082, 24.609, 25.513,
                             26.849, 27.883, 28.749, 29.222, 28.644, 27.979]])

    npt.assert_almost_equal(outliers, outliers_t, decimal=2)


@patch("matplotlib.pyplot.show")
def test_outliers_method(mock_show):
    output = hdr_boxplot(data, threshold=0.95, outliers='forest')
    outliers = output[1]
    outliers_t = np.vstack([[23.740, 23.100, 22.223, 21.972, 22.595, 23.560,
                             25.020, 26.720, 28.196, 29.124, 29.070, 28.397],
                            [25.928, 24.633, 23.177, 22.480, 22.678, 23.325,
                             24.242, 25.189, 25.958, 25.783, 23.995, 22.598],
                            [26.611, 25.935, 24.658, 24.082, 24.609, 25.513,
                             26.849, 27.883, 28.749, 29.222, 28.644, 27.979]])

    npt.assert_almost_equal(outliers, outliers_t, decimal=2)


@patch("matplotlib.pyplot.show")
def test_optimize_bw(mock_show):
    output = hdr_boxplot(data, optimize=True)
    median = output[0]
    median_t = [22.32,  21.17,  20.26,  20.,  20.37,  21.13,  22.28,  24.1,
                25.67,  26.18,  25.26,  23.96]

    npt.assert_almost_equal(median, median_t, decimal=2)


@patch("matplotlib.pyplot.show")
def test_variance(mock_show):
    output = hdr_boxplot(data, variance=0.9)
    median = output[0]
    median_t = [22.49,  21.32,  20.42,  20.21,  20.66,  21.47,  22.6 ,  24.31,
                25.68,  25.96,  24.88,  23.53]

    npt.assert_almost_equal(median, median_t, decimal=2)
