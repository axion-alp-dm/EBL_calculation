import numpy as np
from scipy.stats import norm

class EDispGauss(object):

    def __init__(self, sigma=0.05, bias=0.):
        """

        :param sigma: float or array-like
            energy resolution as percentage,
            if array, then E_reco will have to be provided
            with same dimensions
        :param bias: float or array-like
            energy resolution bias,
            if array, then E_reco will have to be provided
            with same dimensions
        """

        self._sigma = sigma
        self._bias = bias
        self._pdf_matrix = None
        self._e_reco_edges = None
        self._e_true_edges = None

    def fill(self, e_true_edges, e_reco_edges):
        """
        Fill the energy dispoersion matrix

        :param E_true:
        :param E_reco:
        :return:
        """

        e_true_centers = np.sqrt(e_true_edges[1:] * e_true_edges[:-1])
        e_reco_centers = np.sqrt(e_reco_edges[1:] * e_reco_edges[:-1])

        if isinstance(self._bias, float):
            bias = np.ones_like(e_reco_centers) * self._bias

        if isinstance(self._sigma, float):
            sigma = np.ones_like(e_reco_centers) * self._sigma

        pdf = []
        for i, e in enumerate(e_reco_centers):
            pdf.append(norm.pdf(x=e_true_centers,
                                loc=e - bias[i],
                                scale=sigma[i] * e))

        self._pdf_matrix = np.array(pdf).T
        self._pdf_matrix /= self._pdf_matrix.sum(axis=0)
        self._e_reco_edges = e_reco_edges
        self._e_true_edges = e_true_edges

    def plot(self, stretch='log10', ax=None, add_cbar=True, **kwargs):
        """Plot the energy dispersion matrix"""
        kwargs.setdefault('vmin', -5.)
        kwargs.setdefault('vmax', 0.)
        kwargs.setdefault('cmap', 'terrain_r')

        import matplotlib.pyplot as plt

        if stretch =='linear':
            edisp = self._pdf_matrix
            clabel = 'DRM'
        elif stretch =='log10':
            edisp = self._pdf_matrix
            edisp[edisp == 0.] = 1e-40
            edisp = np.log10(self._pdf_matrix)
            clabel ='$\log_{10}(\mathrm{DRM})$'
        else:
            raise ValueError("stretch must either be linear or log10")

        er, et = np.meshgrid(self._e_reco_edges, self._e_true_edges,
                             indexing='ij')
        if ax is None:
            ax = plt.gca()
        im = ax.pcolormesh(et, er, edisp.T, **kwargs)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$E_\mathrm{true}$')
        ax.set_ylabel('$E_\mathrm{reco}$')

        if add_cbar:
            axcbar = plt.colorbar(im, label=clabel)
            return ax, axcbar
        else:
            return ax, None