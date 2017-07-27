import numpy as np
import openturns as ot
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')


def hdr_boxplot(data, alpha=[], threshold=0.9, n_contours=50, plot_data=False):
    """High Density Region boxplot.

    Using the dataset :attr:`data`:

    1. Compute a 2D kernel smoothing with a Gaussian kernel,
    2. Compute contour lines for quartiles 90, 50, 0.001% and :attr:`alpha`,
    3. Plot the bivariate plot,
    4. Compute mediane curve along with quartiles and outliers.

    :param np.array data: dataset (n_samples, n_features)
    :param list(float) alpha: extra contour values
    :param float threshold: threshold for outliers
    :param int n_contours: discretization to compute contour
    :param bool plot_data: append data on the bivariate plot
    :returns: mediane curve along with 50%, 90% quartile (inf and sup curves) and outliers.
    :rtypes: np.array, list(np.array), np.array
    """
    n_sample, dim = data.shape
    # PCA and bivariate plot
    pca = PCA(n_components=2)
    data_r = pca.fit_transform(data)

    print('Explained variance ratio (first two components): {}'
          .format(pca.explained_variance_ratio_))

    plt.figure('Bivariate space')
    plt.scatter(data_r[:, 0], data_r[:, 1], alpha=.8)
    plt.xlabel('First component')
    plt.ylabel('Second component')
    plt.show()

    # Create gaussian kernel
    kernel = ot.KernelSmoothing()
    ks_gaussian = kernel.build(data_r)

    # Evaluate density on a regular grid
    min_max = np.array([data_r.min(axis=0), data_r.max(axis=0)]).T

    x1 = np.linspace(*min_max[0], n_contours)
    x2 = np.linspace(*min_max[1], n_contours)

    contour_grid = np.meshgrid(x1, x2)
    contour_stack = np.dstack(contour_grid).reshape(-1, 2)

    pdf = ks_gaussian.computePDF(contour_stack)
    pdf = np.array(pdf).reshape((n_contours, n_contours))

    X1, X2 = contour_grid

    # Compute contour line of pvalue linked to a given probability level
    alpha.extend([0.9, 0.5, 0.001])
    alpha.sort(reverse=True)

    n_contour_lines = len(alpha)
    pvalues = np.zeros(n_contour_lines)
    for i in range(n_contour_lines):
        levelSet, threshold = ks_gaussian.computeMinimumVolumeLevelSetWithThreshold(alpha[i])
        pvalues[i] = threshold

    print('pvalues: ', pvalues)

    # Create contour plot
    plt.figure('Bivariate space: 2D Kernel Smoothing with Gaussian kernel')
    fig = plt.contour(*contour_grid, pdf, pvalues)

    # Labels: probability instead of density
    fmt = {}
    for i in range(n_contour_lines):
        l = fig.levels[i]
        fmt[l] = "%.0f %%" % (alpha[i] * 100)
    plt.clabel(fig, fig.levels, inline=True, fontsize=10, fmt=fmt)

    if plot_data:
        plt.plot(data_r[:, 0], data_r[:, 1], "b.")
    plt.xlabel('First component')
    plt.ylabel('Second component')

    median = fig.collections[alpha.index(0.001)].get_paths()[0]
    median = np.median(median.vertices, axis=0)
    plt.plot(median[0], median[1], c='r', marker='^')

    plt.show()

    # Find mean, quartiles and outliers curves
    pdf = pdf.flatten()

    # median = pdf.argmax()
    # median = contour_stack[median]
    # plt.plot(median[0], median[1], c='r', marker='^')
    # plt.show()

    pdf_data = np.array(ks_gaussian.computePDF(data_r)).flatten()
    outliers = np.where(pdf_data < threshold)
    outliers = data_r[outliers]

    extreme_quartile = np.where((pdf > pvalues[alpha.index(0.9)]) & (pdf < pvalues[alpha.index(0.5)]))
    extreme_quartile = contour_stack[extreme_quartile]

    mean_quartile = np.where(pdf > pvalues[alpha.index(0.5)])
    mean_quartile = contour_stack[mean_quartile]

    # Inverse transform from bivariate plot to dataset
    median = pca.inverse_transform(median)
    outliers = pca.inverse_transform(outliers)
    extreme_quartile = pca.inverse_transform(extreme_quartile)
    mean_quartile = pca.inverse_transform(mean_quartile)

    extreme_quartile = [extreme_quartile.max(axis=0), extreme_quartile.min(axis=0)]
    mean_quartile = [mean_quartile.max(axis=0), mean_quartile.min(axis=0)]

    plt.figure('Time Serie')
    x_common = np.linspace(1, 12, dim)
    plt.plot(np.array([x_common] * n_sample).T, data.T, alpha=.2)

    plt.fill_between(x_common, *mean_quartile, color='gray', alpha=.4)
    plt.fill_between(x_common, *extreme_quartile, color='gray', alpha=.4)

    plt.plot(x_common, median, c='k')
    plt.plot(np.array([x_common] * len(outliers)).T, outliers.T, c='r', alpha=0.7)

    plt.xlabel('Month of the year')
    plt.ylabel('Water surface temperature (°C)')
    plt.show()

    return median, outliers, extreme_quartile, mean_quartile


# Water surface temperature data from:
# https://www.math.univ-toulouse.fr/~ferraty/SOFTWARES/NPFDA/npfda-datasets.html
# http://www.cpc.ncep.noaa.gov/data/indices/
data = np.loadtxt('data/elnino.dat')
print('Data shape: ', data.shape)

output = hdr_boxplot(data)
output = hdr_boxplot(data, alpha=0.8)
