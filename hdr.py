# import itertools
from statsmodels.compat.python import combinations, range
import numpy as np

from statsmodels.multivariate.pca import PCA
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.graphics import utils


def hdr_boxplot(data, ncomp=2, alpha=[], threshold=0.95, optimize=False,
                n_contours=50, xdata=None, labels=None, ax=None, plot_opts={}):
    """Plot High Density Region boxplot.

    1. Compute a multivariate kernel density estimation,
    2. Compute contour lines for percentiles 90%, 50% and `alpha`%,
    3. Plot the bivariate plot,
    4. Compute mediane curve along with percentiles and outliers curves.

    Parameters
    ----------
    data : sequence of ndarrays or 2-D ndarray
        The vectors of functions to create a functional boxplot from.  If a
        sequence of 1-D arrays, these should all be the same size.
        The first axis is the function index, the second axis the one along
        which the function is defined.  So ``data[0, :]`` is the first
        functional curve.
    ncomp : int, optional
        Number of components to use.  If None, returns the as many as the
        smaller of the number of rows or columns in data.
    alpha : list, optional
        Extra percentile values to compute.
    threshold : float
        Percentile threshold value for outliers detection. High value means
        a lower sensitivity to outliers. Default is `0.95`.
    optimize: bool
        Bandwidth optimization with cross validation or normal inferance
    n_contours : int
        Discretization per dimension of the reduced space.
    xdata : ndarray, optional
        The independent variable for the data.  If not given, it is assumed to
        be an array of integers 0..N with N the length of the vectors in
        `data`.
    labels : sequence of scalar or str, optional
        The labels or identifiers of the curves in `data`.  If given, outliers
        are labeled in the plot.
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    plot_opts : dict, optional
        A dictionary with plotting options.  Any of the following can be
        provided, if not present in `plot_opts` the defaults will be used::

          - 'cmap_outliers', a Matplotlib LinearSegmentedColormap instance.
          - 'c_inner', valid MPL color. Color of the central 50% region
          - 'c_outer', valid MPL color. Color of the non-outlying region
          - 'c_median', valid MPL color. Color of the median.
          - 'lw_outliers', scalar.  Linewidth for drawing outlier curves.
          - 'lw_median', scalar.  Linewidth for drawing the median curve.
          - 'draw_nonout', bool.  If True, also draw non-outlying curves.

    :returns: mediane curve along with 50%, 90% quartile (inf and sup curves)
    and outliers.
    :rtypes: np.array, list(np.array), np.array

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    """
    def _kernel_smoothing(data, optimize=False):
        """Create gaussian kernel.

        Parameters
        ----------
        data : sequence of ndarrays or 2-D ndarray
            The vectors of functions to create a functional boxplot from.  If a
            sequence of 1-D arrays, these should all be the same size.
            The first axis is the function index, the second axis the one along
            which the function is defined.  So ``data[0, :]`` is the first
            functional curve.
        optimize : bool, optional
            Use `normal_reference` or `cv_ml`. Default is False.

        Returns
        -------
        kde : KDEMultivariate instance

        """
        n_samples, dim = data.shape

        if optimize:
            kde = KDEMultivariate(data, bw='cv_ml', var_type='c' * dim)
        else:
            kde = KDEMultivariate(data, bw='normal_reference', var_type='c' * dim)

        return kde

    def _inverse_transform(pca, data):
        """Inverse transform on PCA.

        Use PCA's `project` method by temporary replacing its factors with
        `data`.

        Parameters
        ----------
        pca : statsmodels Principal Component Analysis instance
            The PCA object to use.
        data : sequence of ndarrays or 2-D ndarray
            The vectors of functions to create a functional boxplot from.  If a
            sequence of 1-D arrays, these should all be the same size.
            The first axis is the function index, the second axis the one along
            which the function is defined.  So ``data[0, :]`` is the first
            functional curve.

        Returns
        -------
        projection : array
            nobs by nvar array of the projection onto ncomp factors

        """
        factors = pca.factors
        pca.factors = data.reshape(-1, ncomp)
        projection = pca.project()
        pca.factors = factors
        return projection

    n_samples, dim = data.shape
    # PCA and bivariate plot
    pca = PCA(data, ncomp=ncomp)
    data_r = pca.factors

    # S = pca.eigenvals
    # explained_variance_ = (S ** 2) / (n_samples - 1)
    # total_var = explained_variance_.sum()
    # explained_variance_ratio_ = explained_variance_ / total_var
    # print(explained_variance_[:2], explained_variance_ratio_[:2])

    # Create gaussian kernel
    ks_gaussian = _kernel_smoothing(data_r, optimize)

    # Evaluate density on a regular grid
    min_max = np.array([data_r.min(axis=0), data_r.max(axis=0)]).T
    contour_grid = np.meshgrid(*[np.linspace(*min_max[i], n_contours)
                                 for i in range(ncomp)])
    contour_stack = np.dstack(contour_grid).reshape(-1, ncomp)

    pdf = ks_gaussian.pdf(contour_stack).flatten()

    # Compute contour line of pvalue linked to a given probability level
    alpha.extend([threshold, 0.9, 0.5])  # 0.001
    alpha = list(set(alpha))
    alpha.sort(reverse=True)

    n_percentiles = len(alpha)
    pdf_r = ks_gaussian.pdf(data_r).flatten()
    pvalues = [np.percentile(pdf_r, (1 - alpha[i]) * 100, interpolation='linear')
               for i in range(n_percentiles)]

    # Find mean, quartiles and outliers curves
    median = pdf.argmax()
    median = contour_stack[median]

    outliers = np.where(pdf_r < pvalues[alpha.index(threshold)])
    outliers = data_r[outliers]

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
            extra_quartile = _inverse_transform(pca, extra_quartile)
            extra_quartiles.extend([extra_quartile.max(axis=0),
                                    extra_quartile.min(axis=0)])
    else:
        extra_quartiles = None

    # Inverse transform from bivariate plot to dataset
    median = _inverse_transform(pca, median)[0]
    outliers = _inverse_transform(pca, outliers)
    extreme_quartile = _inverse_transform(pca, extreme_quartile)
    mean_quartile = _inverse_transform(pca, mean_quartile)

    extreme_quartile = [extreme_quartile.max(axis=0), extreme_quartile.min(axis=0)]
    mean_quartile = [mean_quartile.max(axis=0), mean_quartile.min(axis=0)]

    # Plots
    fig, ax = utils.create_mpl_ax(ax)

    # if ncomp == 2:
    #     ax.figure('2D Kernel Smoothing with Gaussian kernel')
    #     contour = ax.contour(*contour_grid,
    #                          pdf.reshape((n_contours, n_contours)), pvalues)
    #     fmt = {}
    #     for i in range(n_percentiles):
    #         lev = contour.levels[i]
    #         fmt[lev] = "%.0f %%" % (alpha[i] * 100)
    #     ax.clabel(contour, contour.levels, inline=True, fontsize=10, fmt=fmt)

    # ax.figure('Bivariate space')
    # ax.tick_params(axis='both', labelsize=8)
    # for i, j in itertools.combinations_with_replacement(range(ncomp), 2):
    #     ax = ax.subplot2grid((ncomp, ncomp), (j, i))
    #     ax.tick_params(axis='both', labelsize=(10 - ncomp))

    #     if i == j:  # diag
    #         x_plot = np.linspace(min(data_r[:, i]), max(data_r[:, i]), 100)[:, np.newaxis]
    #         _ks = _kernel_smoothing(data_r[:, i, np.newaxis], optimize)
    #         ax.plot(x_plot, _ks.pdf(x_plot))
    #     elif i < j:  # lower corners
    #         ax.scatter(data_r[:, i], data_r[:, j], s=5, c='k', marker='o')

    #     if i == 0:
    #         ax.set_ylabel(str(j + 1))
    #     if j == (ncomp - 1):
    #         ax.set_xlabel(str(i + 1))

    if xdata is None:
        xdata = np.arange(0, 1, dim)

    ax.plot(np.array([xdata] * n_samples).T, data.T, alpha=.2)
    ax.fill_between(xdata, *mean_quartile, color='gray', alpha=.4)
    ax.fill_between(xdata, *extreme_quartile, color='gray', alpha=.4)

    try:
        ax.plot(np.array([xdata] * len(extra_quartiles)).T,
                np.array(extra_quartiles).T, color='c', ls='-.', alpha=.4)
    except TypeError:
        pass

    ax.plot(xdata, median, c='k')

    try:
        ax.plot(np.array([xdata] * len(outliers)).T, outliers.T,
                c='r', alpha=0.7)
    except ValueError:
        print('It seems that there are no outliers...')

    return fig, median, outliers, extreme_quartile, mean_quartile, extra_quartiles
