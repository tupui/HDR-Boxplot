from batman.functions import Channel_Flow
from batman.space import Space
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
import numpy as np
import openturns as ot
import pylab as pl

def computeContour2D(X1pars,X2pars,dist):
    X1min,X1max,nX1 = X1pars
    X2min,X2max,nX2 = X2pars
    #
    x1 = np.linspace(X1min, X1max, nX1)
    x2 = np.linspace(X2min, X2max, nX2)
    X1, X2 = np.meshgrid(x1, x2)
    #
    X1flat = X1.flatten()
    X2flat = X2.flatten()
    #
    X1flat = ot.NumericalSample(X1flat,1)
    X2flat = ot.NumericalSample(X2flat,1)
    #
    inputContour = ot.NumericalSample(nX1*nX2, 2)
    inputContour[:,0] = X1flat
    inputContour[:,1] = X2flat
    Z = dist.computePDF(inputContour)
    #
    Z = np.array(Z)
    Z = Z.reshape((nX1,nX2))
    return [X1,X2,Z]

def plotContourByKS2D(data,Ncontour,relativeFactor,alpha,numberOfContourLines,contourByLevelSet,plotData):
    # 1. Creation du noyau gaussien
    noyauGaussien = ot.KernelSmoothing()      # (*)
    ksGaussian = noyauGaussien.build(data) # (*)
    # 2. Evalue la densité sur une grille régulière
    X1min=data[:,0].getMin()[0]
    X1max=data[:,0].getMax()[0]
    X2min=data[:,1].getMin()[0]
    X2max=data[:,1].getMax()[0]
    X1pars = [X1min,X1max,Ncontour]
    X2pars = [X2min,X2max,Ncontour]
    [X1,X2,Z] = computeContour2D(X1pars,X2pars,ksGaussian)
    # TODO MBN Février 2017 : DistributionImplementation.cxx, lignes 2407 dans la méthode computeMinimumVolumeLevelSetWithThreshold
    if (contourByLevelSet):
        # 3. Calcule la ligne de niveau pvalue associée a un niveau donné de probabilité
        numberOfContourLines=len(alpha)
        # TODO MBN Février 2017: levelSet, threshold = ksGaussian.computeMinimumVolumeLevelSetWithThreshold(alpha)
        # computeUnivariateMinimumVolumeLevelSetByQMC ?
        pvalues=np.zeros(numberOfContourLines)
        for i in range(numberOfContourLines):
            levelSet, threshold = ksGaussian.computeMinimumVolumeLevelSetWithThreshold(alpha[i])  # (*)
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
    mydescr = data.getDescription()
    pl.xlabel(mydescr[0])
    pl.ylabel(mydescr[1])

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

output = plotContourByKS2D(ot.Sample(X_r),Ncontour,relativeFactor,alpha,numberOfContourLines,contourByLevelSet,plotData)

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
