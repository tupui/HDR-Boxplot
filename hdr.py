import numpy as np
import openturns as ot
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')


def plot_contour_ks_2d(data, alpha, n_contours=50, level_set=True, plot_data=False):
    """Contour plot of density probability.

    2D kernel smoothing on a dataset with a Gaussian kernel. Then compute
    contour lines using :attr:`alpha`. Finally, plot the bivariate plot.

    :param np.array data: dataset (n_samples, n_features)
    :param list(float) alpha: target quantiles
    :param int n_contours: discretization to compute contour
    :param bool level_set: use OpenTURNS computeMinimumVolumeLevelSetWithThreshold
    :param bool plot_data: append data on the bivariate plot
    :returns: curves associated to alpha list
    :rtypes: np.array
    """
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

    # pvalues = [ 0.00246924,  0.01403278,  0.03061416,  0.03600835]
    # median = [-1, 0]

    # Find quartiles and outliers curves
    pdf = pdf.flatten()

    pdf_data = np.array(ks_gaussian.computePDF(data)).flatten()
    outliers = np.where(pdf_data < pvalues[0])
    outliers = data[outliers]

    extreme_quartile = np.where((pdf > pvalues[0]) & (pdf < pvalues[1]))
    extreme_quartile = contour_stack[extreme_quartile]

    mean_quartile = np.where(pdf > pvalues[1])
    mean_quartile = contour_stack[mean_quartile]

    return median, outliers, extreme_quartile, mean_quartile


# Water surface temperature data from:
# https://www.math.univ-toulouse.fr/~ferraty/SOFTWARES/NPFDA/npfda-datasets.html
# http://www.cpc.ncep.noaa.gov/data/indices/
data = np.loadtxt('data/elnino.dat')
print('Data shape: ', data.shape)

pca = PCA(n_components=2)
X_r = pca.fit_transform(data)

print('Explained variance ratio (first two components): {}'
      .format(pca, pca.explained_variance_ratio_))

plt.figure('Bivariate space')
plt.scatter(X_r[:, 0], X_r[:, 1], alpha=.8)
plt.xlabel('First component')
plt.ylabel('Second component')
plt.show()

output = plot_contour_ks_2d(X_r, [0.9, 0.5, 0.1, 0.001])

median, outliers, extreme_quartile, mean_quartile = output
median = pca.inverse_transform(median)
outliers = pca.inverse_transform(outliers)
extreme_quartile = pca.inverse_transform(extreme_quartile)
mean_quartile = pca.inverse_transform(mean_quartile)

plt.figure('Time Serie')
n_sample, dim = data.shape
x_common = np.linspace(1, 12, dim)
plt.plot(np.array([x_common] * n_sample).T, data.T, alpha=.2)

plt.fill_between(x_common, mean_quartile.max(axis=0),
                 mean_quartile.min(axis=0), color='gray', alpha=.4)
plt.fill_between(x_common, extreme_quartile.max(axis=0),
                 extreme_quartile.min(axis=0), color='gray', alpha=.4)

plt.plot(x_common, median, c='k')
plt.plot(np.array([x_common] * len(outliers)).T, outliers.T, c='r', alpha=0.7)

plt.xlabel('Month of the year')
plt.ylabel('Water surface temperature (°C)')
plt.show()
