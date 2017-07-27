import numpy as np
import openturns as ot
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')


def hdr_boxplot(data, alpha, n_contours=50, level_set=True, plot_data=False):
    """High Density Region boxplot.

    Using the dataset :attr:`data`:

    1. Compute a 2D kernel smoothing with a Gaussian kernel,
    2. Compute contour lines using :attr:`alpha`,
    3. Plot the bivariate plot,
    4. Compute mediane curve along with quartiles and outliers.

    :param np.array data: dataset (n_samples, n_features)
    :param list(float) alpha: target quantiles
    :param int n_contours: discretization to compute contour
    :param bool level_set: use OpenTURNS computeMinimumVolumeLevelSetWithThreshold
    :param bool plot_data: append data on the bivariate plot
    :returns: mediane curve along with 50%, 90% quartile (inf and sup curves) and outliers.
    :rtypes: np.array, list(np.array), np.array
    """
    # PCA and bivariate plot
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    
    print('Explained variance ratio (first two components): {}'
          .format(pca.explained_variance_ratio_))
    
    plt.figure('Bivariate space')
    plt.scatter(data[:, 0], data[:, 1], alpha=.8)
    plt.xlabel('First component')
    plt.ylabel('Second component')
    plt.show()

    # Create gaussian kernel
    kernel = ot.KernelSmoothing()
    ks_gaussian = kernel.build(data)

    # Evaluate density on a regular grid
    min_max = np.array([data.min(axis=0), data.max(axis=0)]).T

    x1 = np.linspace(*min_max[0], n_contours)
    x2 = np.linspace(*min_max[1], n_contours)

    contour_grid = np.meshgrid(x1, x2)
    contour_stack = np.dstack(contour_grid).reshape(-1, 2)

    pdf = ks_gaussian.computePDF(contour_stack)
    pdf = np.array(pdf).reshape((n_contours, n_contours))

    X1, X2 = contour_grid

    plt.figure('Bivariate space: 2D Kernel Smoothing with Gaussian kernel')
    if level_set:
        # Compute contour line of pvalue linked to a given probability level
        n_contour_lines = len(alpha)
        pvalues = np.zeros(n_contour_lines)
        for i in range(n_contour_lines):
            levelSet, threshold = ks_gaussian.computeMinimumVolumeLevelSetWithThreshold(alpha[i])
            pvalues[i] = threshold

        print('pvalues: ', pvalues)

        # Create contour plot
        fig = plt.contour(*contour_grid, pdf, pvalues)

        # Labels: probability instead of density
        fmt = {}
        for i in range(n_contour_lines):
            l = fig.levels[i]
            fmt[l] = "%.0f %%" % (alpha[i] * 100)
        plt.clabel(fig, fig.levels, inline=True, fontsize=10, fmt=fmt)
    else:
        # Create contour plot
        fig = plt.contour(*contour_grid, pdf, n_contour_lines)

        fmt = {}
        for l in fig.levels:
            fmt[l] = "%.0e" % (l)
        plt.clabel(fig, fig.levels, inline=True, fontsize=10, fmt=fmt)

    if plot_data:
        plt.plot(data[:, 0], data[:, 1], "b.")
    plt.xlabel('First component')
    plt.ylabel('Second component')

    # median = np.unravel_index(pdf.argmax(), pdf.shape)
    # median = (X1[median], X2[median])
    # plt.plot(median, c='r', marker='^')

    median_path = fig.collections[-1].get_paths()[0]
    median = np.median(median_path.vertices, axis=0)
    plt.plot(median[0], median[1], c='r', marker='^')

    plt.show()

    # Find quartiles and outliers curves
    pdf = pdf.flatten()

    pdf_data = np.array(ks_gaussian.computePDF(data)).flatten()
    outliers = np.where(pdf_data < pvalues[0])
    outliers = data[outliers]

    extreme_quartile = np.where((pdf > pvalues[0]) & (pdf < pvalues[1]))
    extreme_quartile = contour_stack[extreme_quartile]

    mean_quartile = np.where(pdf > pvalues[1])
    mean_quartile = contour_stack[mean_quartile]

    # Inverse transform from bivariate plot to dataset
    median = pca.inverse_transform(median)
    outliers = pca.inverse_transform(outliers)
    extreme_quartile = pca.inverse_transform(extreme_quartile)
    mean_quartile = pca.inverse_transform(mean_quartile)

    extreme_quartile = [extreme_quartile.max(axis=0), extreme_quartile.min(axis=0)]
    mean_quartile = [mean_quartile.max(axis=0), mean_quartile.min(axis=0)]

    plt.figure('Time Serie')
    n_sample, dim = data.shape
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

output = hdr_boxplot(data, [0.9, 0.5, 0.1, 0.001])
