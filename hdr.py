import numpy as np

from statsmodels.multivariate.pca import PCA
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.graphics import utils
from statsmodels.compat.python import range


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
    _, dim = data.shape

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
    pca.factors = data.reshape(-1, factors.shape[1])
    projection = pca.project()
    pca.factors = factors
    return projection


def hdrboxplot(data, ncomp=2, alpha=None, threshold=0.95, optimize=False,
               n_contours=50, xdata=None, labels=None, ax=None):
    """Plot High Density Region boxplot.

    1. Compute a multivariate kernel density estimation,
    2. Compute contour lines for quantiles 90%, 50% and `alpha`%,
    3. Plot the bivariate plot,
    4. Compute mediane curve along with quantiles and outliers curves.

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
    alpha : list of floats between 0 and 1, optional
        Extra quantile values to compute. Default is None
    threshold : float between 0 and 1
        Percentile threshold value for outliers detection. High value means
        a lower sensitivity to outliers. Default is `0.95`.
    optimize: bool
        Bandwidth optimization with cross validation or normal inferance
    n_contours : int
        Discretization per dimension of the reduced space.
    xdata : ndarray, optional
        The independent variable for the data. If not given, it is assumed to
        be an array of integers 0..N with N the length of the vectors in
        `data`.
    labels : sequence of scalar or str, optional
        The labels or identifiers of the curves in `data`. If given, outliers
        are labeled in the plot.
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.
    hdr_res : dict

         - 'median', array. Median curve.
         - 'mean_quantile', array. 50% quantile band. [sup, inf] curves
         - 'extreme_quantile', list of array. 90% quantile band. [sup, inf]
            curves.
         - 'extra_quantiles', list of array. Extra quantile band.
            [sup, inf] curves.
         - 'outliers', ndarray. Outlier curves.

    See Also
    --------
    banddepth, rainbowplot, fboxplot

    Notes
    -----
    The median curve is the curve with the highest probability on the reduced
    space of a Principal Component Analysis (PCA).

    Outliers are defined as curves that fall outside the band corresponding
    to the quantile given by `threshold`.

    The non-outlying region is defined as the band made up of all the
    non-outlying curves.

    Behind the scene, the dataset is represented as a matrix. Each line
    corresponding to a 1D curve. This matrix is then decomposed using Principal
    Components Analysis (PCA). This allows to represent the data using a finite
    number of modes, or components. This compression process allows to turn the
    functional representation into a scalar representation of the matrix. In
    other words, you can visualize each curve from its components. Each curve
    is thus a point in this reduced space. With 2 components, this is called a
    bivariate plot (2D plot).

    In this plot, if some points are adjacent (similar components), it means
    that back in the original space, the curves are similar. Then, finding the
    median curve means finding the higher density region (HDR) in the reduced
    space. Moreover, the more you get away from this HDR, the more the curve is
    unlikely to be similar to the other curves.

    Using a kernel smoothing technique, the probability density function (PDF)
    of the multivariate space can be recover. From this PDF, it is possible to
    compute the density probability linked to the cluster of points and plot
    its contours.

    Finally, using these contours, the different quantiles can be extracted
    allong with the median curve and the outliers.

    References
    ----------
    [1] R.J. Hyndman and H.L. Shang, "Rainbow Plots, Bagplots, and Boxplots for
        Functional Data", vol. 19, pp. 29-45, 2010.

    Examples
    --------
    Load the El Nino dataset.  Consists of 60 years worth of Pacific Ocean sea
    surface temperature data.

    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.elnino.load()

    Create a functional boxplot.  We see that the years 1982-83 and 1997-98 are
    outliers; these are the years where El Nino (a climate pattern
    characterized by warming up of the sea surface and higher air pressures)
    occurred with unusual intensity.

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> res = sm.graphics.hdrboxplot(data.raw_data[:, 1:],
    ...                              labels=data.raw_data[:, 0].astype(int),
    ...                              ax=ax)

    >>> ax.set_xlabel("Month of the year")
    >>> ax.set_ylabel("Sea surface temperature (C)")
    >>> ax.set_xticks(np.arange(13, step=3) - 1)
    >>> ax.set_xticklabels(["", "Mar", "Jun", "Sep", "Dec"])
    >>> ax.set_xlim([-0.2, 11.2])

    >>> plt.show()

    .. plot:: plots/graphics_functional_hdrboxplot.py

    """
    fig, ax = utils.create_mpl_ax(ax)

    data = np.asarray(data)
    if xdata is None:
        xdata = np.arange(data.shape[1])

    n_samples, dim = data.shape
    # PCA and bivariate plot
    pca = PCA(data, ncomp=ncomp)
    data_r = pca.factors

    # Create gaussian kernel
    ks_gaussian = _kernel_smoothing(data_r, optimize)

    # Evaluate density on a regular grid
    min_max = np.array([data_r.min(axis=0), data_r.max(axis=0)]).T
    contour_grid = np.meshgrid(*[np.linspace(min_max[i, 0],
                                             min_max[i, 1],
                                             n_contours)
                                 for i in range(ncomp)])
    contour_stack = np.dstack(contour_grid).reshape(-1, ncomp)

    pdf = ks_gaussian.pdf(contour_stack).flatten()

    # Compute contour line of pvalue linked to a given probability level
    if alpha is None:
        alpha = [threshold, 0.9, 0.5]
    else:
        alpha.extend([threshold, 0.9, 0.5])
        alpha = list(set(alpha))
        alpha.sort(reverse=True)

    n_quantiles = len(alpha)
    pdf_r = ks_gaussian.pdf(data_r).flatten()
    pvalues = [np.quantile(pdf_r, (1 - alpha[i]) * 100,
                           interpolation='linear')
               for i in range(n_quantiles)]

    # Find mean, quartiles and outliers curves
    median = pdf.argmax()
    median = contour_stack[median]

    outliers = np.where(pdf_r < pvalues[alpha.index(threshold)])
    labels = labels[outliers]
    outliers = data[outliers]

    extreme_quantile = np.where((pdf > pvalues[alpha.index(0.9)])
                                & (pdf < pvalues[alpha.index(0.5)]))
    extreme_quantile = contour_stack[extreme_quantile]

    mean_quantile = np.where(pdf > pvalues[alpha.index(0.5)])
    mean_quantile = contour_stack[mean_quantile]

    extra_alpha = [i for i in alpha
                   if 0.5 != i and 0.9 != i and threshold != i]
    if extra_alpha != []:
        extra_quantiles = []
        for i in extra_alpha:
            extra_quantile = np.where(pdf > pvalues[alpha.index(i)])
            extra_quantile = contour_stack[extra_quantile]
            extra_quantile = _inverse_transform(pca, extra_quantile)
            extra_quantiles.extend([extra_quantile.max(axis=0),
                                    extra_quantile.min(axis=0)])
    else:
        extra_quantiles = None

    # Inverse transform from bivariate plot to dataset
    median = _inverse_transform(pca, median)[0]
    extreme_quantile = _inverse_transform(pca, extreme_quantile)
    mean_quantile = _inverse_transform(pca, mean_quantile)

    extreme_quantile = [extreme_quantile.max(axis=0),
                        extreme_quantile.min(axis=0)]
    mean_quantile = [mean_quantile.max(axis=0),
                     mean_quantile.min(axis=0)]

    hdr_res = {
        "median": median,
        "mean_quantile": mean_quantile,
        "extreme_quantile": extreme_quantile,
        "extra_quantiles": extra_quantiles,
        "outliers": outliers
    }

    # Plots
    ax.plot(np.array([xdata] * n_samples).T, data.T,
            c='c', alpha=.1, label='dataset')
    ax.plot(xdata, median, c='k', label='Median')
    ax.fill_between(xdata, *mean_quantile,
                    color='gray', alpha=.4,  label='50th quantile')
    ax.fill_between(xdata, *extreme_quantile,
                    color='gray', alpha=.3, label='90th quantile')

    if len(extra_quantiles) != 0:
        ax.plot(np.array([xdata] * len(extra_quantiles)).T,
                np.array(extra_quantiles).T,
                c='y', ls='-.', alpha=.4, label='Extra quantiles')

    if len(outliers) != 0:
        for ii, outlier in enumerate(outliers):
            label = str(labels[ii]) if labels is not None else None
            ax.plot(xdata, outlier,
                    ls='--', alpha=0.7, label=label)

    if labels is not None:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

    return fig, hdr_res
