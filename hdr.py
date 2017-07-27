import numpy as np
import openturns as ot
import pylab as pl
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

def contour_2d(data, n_contours):
    """Compute contour in 2D.

    :param np.array data: dataset (n_samples, n_features)
    :param int n_contours: discretization for the contours
    :return: contour grid, pdf values and gaussian kernel
    """
    # Create gaussian kernel
    # ot.Sample(data)
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

    return contour_grid, pdf, ks_gaussian

def plotContourByKS2D(data,Ncontour,relativeFactor,alpha,numberOfContourLines,contourByLevelSet,plotData):
    
    contour_grid, Z, ks_gaussian = contour_2d(data, Ncontour)

    X1, X2 = contour_grid
    
    if (contourByLevelSet):
        # 3. Calcule la ligne de niveau pvalue associée a un niveau donné de probabilité
        numberOfContourLines=len(alpha)
        pvalues=np.zeros(numberOfContourLines)
        for i in range(numberOfContourLines):
            levelSet, threshold = ks_gaussian.computeMinimumVolumeLevelSetWithThreshold(alpha[i])  # (*)
            pvalues[i]=threshold
        # 4. Cree le contour
        print(pvalues)
        CS = pl.contour(X1, X2, Z, pvalues)
        # 5. Calcule les labels : affiche la probabilité plutôt que la densité
        fmt = {}
        for i in range(numberOfContourLines):
            l = CS.levels[i]
            fmt[l] = "%.0f %%" % (alpha[i]*100)
        # 6. Create contour plot (enfin !)
        pl.clabel(CS, CS.levels, inline=True, fontsize=10, fmt=fmt)
    else:
        # 4. Cree le contour
        CS = pl.contour(X1, X2, Z, numberOfContourLines)
        #
        fmt = {}
        for l in CS.levels:
            fmt[l] = "%.0e" % (l)
        # 6. Create contour plot (enfin !)
        pl.clabel(CS, CS.levels, inline=True, fontsize=10, fmt=fmt)
    # 7. Dessine le nuage
    if (plotData):
        pl.plot(data[:,0],data[:,1],"b.")
    pl.title('2D Kernel Smoothing with Gaussian kernel')
    pl.xlabel('First component')
    pl.ylabel('Second component')

    # median = np.unravel_index(Z.argmax(), Z.shape)
    # median = (X1[median], X2[median])
    # pl.plot(median, c='r', marker='^')

    median_path = CS.collections[-1].get_paths()[0]
    median = np.median(median_path.vertices, axis=0)
    pl.plot(median[0], median[1], c='r', marker='^')

    # pl.contourf(X1, X2, Z)

    pl.show()


    outlier_path = np.unravel_index(np.where(Z < pvalues[0]), Z.shape)
    outlier_path = np.array([X1[outlier_path], X2[outlier_path]])
    extreme_quartile_path = np.unravel_index(np.where((Z > pvalues[0]) & (Z < pvalues[1])), Z.shape)
    mean_quartile_path = np.unravel_index(np.where((Z > pvalues[1]) & (Z < pvalues[2])), Z.shape)

    return median, outlier_path, extreme_quartile_path, mean_quartile_path

# https://www.math.univ-toulouse.fr/~ferraty/SOFTWARES/NPFDA/npfda-datasets.html
# http://www.cpc.ncep.noaa.gov/data/indices/
data = np.loadtxt('data/elnino.dat')

print(data.shape)

pca = PCA(n_components=2)
print(pca)

X_r = pca.fit_transform(data)

print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
plt.scatter(X_r[:, 0], X_r[:, 1], alpha=.8)
plt.show()


Ncontour=50
relativeFactor = 0.1
alpha=[0.9,0.5,0.1, 0.001]
numberOfContourLines = 5
contourByLevelSet = True
plotData = False

output = plotContourByKS2D(X_r,Ncontour,relativeFactor,alpha,numberOfContourLines,contourByLevelSet,plotData)

median, outlier_path, extreme_quartile_path, mean_quartile_path = output
median = pca.inverse_transform(median)
outlier_path = pca.inverse_transform(outlier_path)
extreme_quartile_path = pca.inverse_transform(extreme_quartile_path)
mean_quartile_path = pca.inverse_transform(mean_quartile_path)


plt.figure()
n_sample, dim = data.shape
x_common = np.linspace(1, 10, dim)
plt.plot(np.array([x_common] * n_sample).T, data.T, alpha=.3)
plt.plot(x_common, median, c='k')

plt.plot(np.array([x_common] * len(outlier_path)).T, outlier_path.T, c='r')

plt.show()
