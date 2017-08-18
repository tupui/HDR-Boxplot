"""
High Density Region Boxplot
---------------------------
"""
from logzero import logger
import itertools
from multiprocessing import Pool
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
np.set_printoptions(precision=3)


def kernel_smoothing(data, optimize=False):
    """Create gaussian kernel.

    :param bool optimize: use global optimization of grid search
    :return: gaussian kernel
    :rtype: :class:`sklearn.neighbors.KernelDensity`
    """
    n_samples, dim = data.shape
    cv = n_samples if n_samples < 50 else 50

    if optimize:
        def bw_score(bw):
            score = cross_val_score(KernelDensity(bandwidth=bw),
                                    data, cv=cv, n_jobs=-1)
            return - score.mean()

        bounds = [(0., 5.)]
        results = differential_evolution(bw_score, bounds, maxiter=5)
        bw = results.x
        ks_gaussian = KernelDensity(bandwidth=bw)
        ks_gaussian.fit(data)
    else:
        scott = n_samples ** (-1. / (dim + 4))
        silverman = (n_samples * (dim + 2) / 4.) ** (-1. / (dim + 4))
        bandwidth = np.hstack([np.linspace(0.1, 5.0, 30), scott, silverman])
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': bandwidth},
                            cv=cv, n_jobs=-1)  # 20-fold cross-validation
        grid.fit(data)
        ks_gaussian = grid.best_estimator_

    return ks_gaussian


class HdrBoxplot:

    """High Density Region boxplot.

    From a given dataset, it computes the HDR-boxplot. Results are accessibles
    directly through class attributes:

    - :attr:`median` : median curve,
    - :attr:`outliers` : outliers regarding a given threshold,
    - :attr:`extreme_percentile` : 90% percentile band,
    - :attr:`extra_percentiles` : other percentile bands,
    - :attr:`mean_percentile` : 50 % percentile band.

    The following methods are for convenience:

    - :func:`HdrBoxplot.plot`

    :Example:

    ::

        >> hdr = HdrBoxplot(data)
        >> hdr.plot()

    """

    def __init__(self, data, variance=0.8, alpha=[],
                 threshold=0.95, outliers_method='kde', optimize=False,
                 n_contours=50):
        """Compute HDR Boxplot on :attr:`data`.

        1. Compute a 2D kernel smoothing with a Gaussian kernel,
        2. Compute contour lines for percentiles 90, 50 and :attr:`alpha`,
        3. Compute mediane curve along with percentiles regions and outlier
        curves.

        :param np.array data: dataset (n_samples, n_features)
        :param float variance: percentage of total variance to conserve
        :param array_like alpha: extra contour values
        :param float threshold: threshold for outliers
        :param str outliers_method: detection method ['kde', 'forest']
        :param bool optimize: bandwidth global optimization or grid search
        :param int n_contours: discretization to compute contour
        """
        self.data = data
        self.threshold = threshold
        self.outliers_method = outliers_method
        self.optimize = optimize
        self.n_contours = n_contours

        self.n_samples, self.dim = self.data.shape
        logger.info('Dataset with:\n-> {} samples\n-> {} features'
                    .format(self.n_samples, self.dim))
        # PCA and bivariate plot
        self.pca = PCA(n_components=variance, svd_solver='full')
        self.data_r = self.pca.fit_transform(self.data)
        self.n_components = len(self.pca.explained_variance_ratio_)

        logger.info('Explained variance ratio: {} -> {:0.3f}'
                    .format(self.pca.explained_variance_ratio_,
                            np.sum(self.pca.explained_variance_ratio_)))

        # Create gaussian kernel
        self.ks_gaussian = kernel_smoothing(self.data_r, self.optimize)

        # Boundaries of the n-variate space
        self.bounds = np.array([self.data_r.min(axis=0), self.data_r.max(axis=0)]).T

        # Create list of percentile values
        alpha.extend([threshold, 0.9, 0.5])
        alpha = list(set(alpha))
        alpha.sort(reverse=True)
        self.alpha = alpha
        logger.debug('alpha: {}'.format(self.alpha))
        self.n_alpha = len(self.alpha)

        # Compute PDF values associated to each percentile
        self.pdf_r = np.exp(self.ks_gaussian.score_samples(self.data_r)).flatten()
        self.pvalues = np.array([np.percentile(self.pdf_r, (1 - self.alpha[i]) * 100,
                                               interpolation='linear')
                                 for i in range(self.n_alpha)])

        logger.debug('pvalues: {}'.format(self.pvalues))

        def pdf(x):
            """Compute -PDF given components."""
            return - np.exp(self.ks_gaussian.score_samples(x.reshape(1, -1)))

        # Find mean, percentiles and outliers curves
        results = differential_evolution(pdf, bounds=self.bounds, maxiter=5)
        median_r = results.x

        outliers = self.find_outliers(data=self.data_r, pdf=self.pdf_r,
                                      method=self.outliers_method,
                                      threshold=self.threshold)

        extra_alpha = [i for i in self.alpha if 0.5 != i and 0.9 != i and threshold != i]
        if extra_alpha != []:
            self.extra_percentiles = [y for x in extra_alpha
                                      for y in self.band_percentiles([x])]
        else:
            self.extra_percentiles = None

        # Inverse transform from n-variate plot to original dataset's shape
        self.median = self.pca.inverse_transform(median_r)
        self.outliers = self.pca.inverse_transform(outliers)
        self.extreme_percentile = self.band_percentiles([0.9, 0.5])
        self.mean_percentile = self.band_percentiles([0.5])

    def band_percentiles(self, band):
        """Find extreme curves for a percentile band.

        From the :attr:`band` of percentiles, the associated PDF extrema values
        are computed. If `min_alpha` is not provided (single percentile value),
        `max_pdf` is set to `1E6` in order not to constrain the problem on high
        values.

        An optimization is performed per component in order to find the min and
        max curves. This is done by comparing the PDF value of a given curve
        with the band PDF.

        :param array_like band: alpha values `[max_alpha, min_alpha]` ex: [0.9, 0.5]
        :return: `[max_percentile, min_percentile]` (2, n_features)
        :rtype: list(array_like)
        """
        min_pdf = self.pvalues[self.alpha.index(band[0])]
        try:
            max_pdf = self.pvalues[self.alpha.index(band[1])]
        except IndexError:
            max_pdf = 1E6
        self.band = [min_pdf, max_pdf]

        with Pool() as pool:
            band_percentiles = pool.map(self._min_max_band, range(self.dim))

        band_percentiles = list(zip(*band_percentiles))

        return band_percentiles

    def _curve_constrain(self, x, idx, sign):
        """Find out if the curve is within the band.

        The curve value at :attr:`idx` for a given PDF is only returned if
        within bounds defined by the band. Otherwise, 1E6 is returned.

        :param float x: curve in reduced space
        :param int idx: index value of the components to compute
        :param int sign: return positive or negative value
        :return: Curve value at :attr:`idx`
        :rtype: float
        """
        x = x.reshape(1, -1)
        pdf = np.exp(self.ks_gaussian.score_samples(x))
        if self.band[0] < pdf < self.band[1]:
            value = sign * self.pca.inverse_transform(x)[0][idx]
        else:
            value = 1E6
        return value

    def _min_max_band(self, idx):
        """Min an max values at :attr:`idx`.

        Global optimization to find the extrema per component.

        :param int idx: curve index
        :returns: [max, min] curve values at :attr:`idx`
        :rtype: tuple(float)
        """
        max_ = differential_evolution(self._curve_constrain, bounds=self.bounds,
                                      args=(idx, -1, ),
                                      maxiter=7)
        min_ = differential_evolution(self._curve_constrain, bounds=self.bounds,
                                      args=(idx, 1, ),
                                      maxiter=7)
        return (self.pca.inverse_transform(max_.x)[idx],
                self.pca.inverse_transform(min_.x)[idx])

    def find_outliers(self, data, pdf=None, method='kde', threshold=0.95):
        """Detect outliers.

        The *Isolation forrest* method requires additional computations to find
        the centroide. This operation is only performed once and stored in
        :attr:`self.detector`. Thus calling, several times the method will not
        cause any overhead.

        :param np.array data: data from which to extract outliers
        :param np.array pdf: pdf values to examine
        :param str method: detection method ['kde', 'forest']
        :param float threshold: detection sensitivity
        """
        if method == 'kde':
            outliers = np.where(pdf < self.pvalues[self.alpha.index(threshold)])
            outliers = data[outliers]
        elif method == 'forest':
            try:
                outliers = np.where(self.detector.predict(data) == -1)
            except AttributeError:
                forrest = IsolationForest(contamination=(1 - threshold), n_jobs=-1)
                self.detector = forrest.fit(data)
                outliers = np.where(self.detector.predict(data) == -1)
            outliers = data[outliers]
        else:
            logger.error('Unknown outlier method: no detection')
            outliers = []

        return outliers

    def plot(self, samples=None, fname=None, x_common=None, xlabel='t', ylabel='y'):
        """Functional plot and n-variate space.

        If :attr:`self.n_components` is 2, an additional contour plot is done.
        If :attr:`samples` is `None`, the dataset is used for all plots ;
        otherwize the given sample is used.

        :param array_like, shape (n_samples, n_features): samples to plot
        :param str fname: wether to export to filename or display the figures
        :param array_like, shape (1, n_features) x_common: abscissa
        :param str xlabel: label for x axis
        :param str ylabel: label for y axis
        """
        figures = []
        if samples is not None:
            data = samples
            data_r = self.pca.fit_transform(data)
            n_samples = len(data)
        else:
            data = self.data
            data_r = self.data_r
            n_samples = self.n_samples

        if self.n_components == 2:
            contour_grid = np.meshgrid(*[np.linspace(*self.bounds[i], self.n_contours)
                                         for i in range(self.n_components)])
            contour_stack = np.dstack(contour_grid).reshape(-1, self.n_components)
            pdf = np.exp(self.ks_gaussian.score_samples(contour_stack)).flatten()

            figures.append(plt.figure('2D Kernel Smoothing with Gaussian kernel'))
            contour = plt.contour(*contour_grid,
                                  pdf.reshape((self.n_contours, self.n_contours)),
                                  self.pvalues)
            fmt = {}
            for i in range(self.n_alpha):
                lev = contour.levels[i]
                fmt[lev] = "%.0f %%" % (self.alpha[i] * 100)
            plt.clabel(contour, contour.levels, inline=True, fontsize=10, fmt=fmt)

        figures.append(plt.figure('Bivariate space'))
        plt.tick_params(axis='both', labelsize=8)
        for i, j in itertools.combinations_with_replacement(range(self.n_components), 2):
            ax = plt.subplot2grid((self.n_components, self.n_components), (j, i))
            ax.tick_params(axis='both', labelsize=(10 - self.n_components))

            if i == j:  # diag
                x_plot = np.linspace(min(data_r[:, i]),
                                     max(data_r[:, i]), 100)[:, np.newaxis]
                _ks = kernel_smoothing(data_r[:, i, np.newaxis], self.optimize)
                ax.plot(x_plot, np.exp(_ks.score_samples(x_plot)))
            elif i < j:  # lower corners
                ax.scatter(data_r[:, i], data_r[:, j], s=5, c='k', marker='o')

            if i == 0:
                ax.set_ylabel(str(j + 1))
            if j == (self.n_components - 1):
                ax.set_xlabel(str(i + 1))

        figures.append(plt.figure('Time Serie'))
        if x_common is None:
            x_common = np.linspace(0, 1, self.dim)
        plt.plot(np.array([x_common] * n_samples).T, data.T,
                 c='c', alpha=.1, label='dataset')
        plt.fill_between(x_common, *self.mean_percentile,
                         color='gray', alpha=.4,  label='50th percentile')
        plt.fill_between(x_common, *self.extreme_percentile,
                         color='gray', alpha=.3, label='90th percentile')

        try:
            plt.plot(np.array([x_common] * len(self.extra_percentiles)).T,
                     np.array(self.extra_percentiles).T,
                     c='y', ls='-.', alpha=.4, label='Extra percentiles')
        except TypeError:
            pass

        plt.plot(x_common, self.median, c='k', label='Median')

        try:
            plt.plot(np.array([x_common] * len(self.outliers)).T, self.outliers.T,
                     c='r', ls='--', alpha=0.7, label='Outliers')
        except ValueError:
            logger.debug('It seems that there are no outliers...')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')

        try:
            pdf = matplotlib.backends.backend_pdf.PdfPages(fname)
            for fig in figures:
                pdf.savefig(fig, transparent=True, bbox_inches='tight')
            pdf.close()
        except ValueError:
            plt.show()
            plt.close('all')
