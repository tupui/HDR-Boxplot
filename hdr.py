import os
import itertools
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from statsmodels.nonparametric.kernel_density import KDEMultivariate

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
plt.switch_backend('Agg')


def hdr_boxplot(data, xdata=None, path=None, variance=0.8, alpha=[],
                threshold=0.95, outliers='kde', optimize=False,
                n_contours=50, xlabel='t', ylabel='y'):
    """High Density Region boxplot.

    Using the dataset :attr:`data`:

    1. Compute a 2D kernel smoothing with a Gaussian kernel,
    2. Compute contour lines for quartiles 90, 50 and :attr:`alpha`,
    3. Plot the bivariate plot,
    4. Compute mediane curve along with quartiles and outliers.

    Parameters
    ----------
    data : sequence of ndarrays or 2-D ndarray
        The vectors of functions to create a functional boxplot from.  If a
        sequence of 1-D arrays, these should all be the same size.
        The first axis is the function index, the second axis the one along
        which the function is defined.  So ``data[0, :]`` is the first
        functional curve.
    xdata : ndarray, optional
        The independent variable for the data.  If not given, it is assumed to
        be an array of integers 0..N with N the length of the vectors in
        `data`.
    :param float variance: percentage of total variance to conserve
    :param list(float) alpha: extra contour values
    :param float threshold: threshold for outliers
    :param str outliers: detection method ['kde', 'forest']
    :param bool optimize: bandwidth global optimization or grid search
    :param int n_contours: discretization to compute contour
    :param str xlabel: label for x axis
    :param str ylabel: label for y axis
    :returns: mediane curve along with 50%, 90% quartile (inf and sup curves)
    and outliers.
    :rtypes: np.array, list(np.array), np.array
    """

    def _kernel_smoothing(data, optimize=False):
        """Create gaussian kernel."""
        n_sample, dim = data.shape

        if optimize:
            kde = KDEMultivariate(data, bw='cv_ml', var_type='c' * dim)
        else:
            kde = KDEMultivariate(data, bw='normal_reference', var_type='c' * dim)

        return kde

    n_sample, dim = data.shape
    # PCA and bivariate plot
    pca = PCA(n_components=variance, svd_solver='full')
    data_r = pca.fit_transform(data)
    n_components = len(pca.explained_variance_ratio_)

    print('Explained variance ratio: {} -> {}'
          .format(pca.explained_variance_ratio_,
                  np.sum(pca.explained_variance_ratio_)))

    # Create gaussian kernel
    ks_gaussian = _kernel_smoothing(data_r, optimize)

    # Evaluate density on a regular grid
    min_max = np.array([data_r.min(axis=0), data_r.max(axis=0)]).T
    contour_grid = np.meshgrid(*[np.linspace(*min_max[i], n_contours)
                                 for i in range(n_components)])
    contour_stack = np.dstack(contour_grid).reshape(-1, n_components)

    pdf = ks_gaussian.pdf(contour_stack).flatten()

    # Compute contour line of pvalue linked to a given probability level
    alpha.extend([threshold, 0.9, 0.5])  # 0.001
    alpha = list(set(alpha))
    alpha.sort(reverse=True)
    print('alpha: ', alpha)

    n_contour_lines = len(alpha)
    pdf_r = ks_gaussian.pdf(data_r).flatten()
    pvalues = [np.percentile(pdf_r, (1 - alpha[i]) * 100, interpolation='linear')
               for i in range(n_contour_lines)]

    print('pvalues: ', pvalues)

    # Find mean, quartiles and outliers curves
    median_r = pdf.argmax()
    median_r = contour_stack[median_r]

    if outliers == 'kde':
        outliers = np.where(pdf_r < pvalues[alpha.index(threshold)])
        outliers = data_r[outliers]
    elif outliers == 'forest':
        forrest = IsolationForest(contamination=(1 - threshold), n_jobs=-1)
        detector = forrest.fit(data_r)
        outliers = np.where(detector.predict(data_r) == -1)
        outliers = data_r[outliers]
    else:
        print('Unknown outlier method: no detection')
        outliers = []

    extreme_quartile = np.where((pdf > pvalues[alpha.index(0.9)])
                                & (pdf < pvalues[alpha.index(0.5)]))
    extreme_quartile = contour_stack[extreme_quartile]

    mean_quartile = np.where(pdf > pvalues[alpha.index(0.5)])
    mean_quartile = contour_stack[mean_quartile]

    extra_alpha = [i for i in alpha if 0.5 != i and 0.9 != i and threshold != i]
    if extra_alpha != []:
        extra_quartiles = []
        for i in extra_alpha:
            extra_quartile = np.where(pdf > pvalues[alpha.index(i)])
            extra_quartile = contour_stack[extra_quartile]
            extra_quartile = pca.inverse_transform(extra_quartile)
            extra_quartiles.extend([extra_quartile.max(axis=0),
                                    extra_quartile.min(axis=0)])
    else:
        extra_quartiles = None

    # Inverse transform from bivariate plot to dataset
    median = pca.inverse_transform(median_r)
    outliers = pca.inverse_transform(outliers)
    extreme_quartile = pca.inverse_transform(extreme_quartile)
    mean_quartile = pca.inverse_transform(mean_quartile)

    extreme_quartile = [extreme_quartile.max(axis=0), extreme_quartile.min(axis=0)]
    mean_quartile = [mean_quartile.max(axis=0), mean_quartile.min(axis=0)]

    # Plots
    figures = []

    if n_components == 2:
        figures.append(plt.figure('2D Kernel Smoothing with Gaussian kernel'))
        contour = plt.contour(*contour_grid,
                              pdf.reshape((n_contours, n_contours)), pvalues)
        # contour = plt.contourf(*contour_grid,
        #                        pdf.reshape((n_contours, n_contours)), 100)
        # plt.colorbar(contour, shrink=0.8, extend='both')
        # Labels: probability instead of density
        fmt = {}
        for i in range(n_contour_lines):
            lev = contour.levels[i]
            fmt[lev] = "%.0f %%" % (alpha[i] * 100)
        plt.clabel(contour, contour.levels, inline=True, fontsize=10, fmt=fmt)

    figures.append(plt.figure('Bivariate space'))
    plt.tick_params(axis='both', labelsize=8)
    for i, j in itertools.combinations_with_replacement(range(n_components), 2):
        ax = plt.subplot2grid((n_components, n_components), (j, i))
        ax.tick_params(axis='both', labelsize=(10 - n_components))

        if i == j:  # diag
            x_plot = np.linspace(min(data_r[:, i]), max(data_r[:, i]), 100)[:, np.newaxis]
            _ks = _kernel_smoothing(data_r[:, i, np.newaxis], optimize)
            ax.plot(x_plot, _ks.pdf(x_plot))
        elif i < j:  # lower corners
            ax.scatter(data_r[:, i], data_r[:, j], s=5, c='k', marker='o')

        if i == 0:
            ax.set_ylabel(str(j + 1))
        if j == (n_components - 1):
            ax.set_xlabel(str(i + 1))

    figures.append(plt.figure('Time Serie'))
    if xdata is None:
        xdata = np.arange(0, 1, dim)
    plt.plot(np.array([xdata] * n_sample).T, data.T, alpha=.2)
    plt.fill_between(xdata, *mean_quartile, color='gray', alpha=.4)
    plt.fill_between(xdata, *extreme_quartile, color='gray', alpha=.4)

    try:
        plt.plot(np.array([xdata] * len(extra_quartiles)).T,
                 np.array(extra_quartiles).T, color='c', ls='-.', alpha=.4)
    except TypeError:
        pass

    plt.plot(xdata, median, c='k')

    try:
        plt.plot(np.array([xdata] * len(outliers)).T, outliers.T,
                 c='r', alpha=0.7)
    except ValueError:
        print('It seems that there are no outliers...')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    try:
        path = os.path.join(path, 'hdr_boxplot.pdf')
        pdf = matplotlib.backends.backend_pdf.PdfPages(path)
        for fig in figures:
            fig.tight_layout()
            pdf.savefig(fig, transparent=True, bbox_inches='tight')
        pdf.close()
    except TypeError:
        plt.show()
        plt.close('all')

    return median, outliers, extreme_quartile, mean_quartile, extra_quartiles
