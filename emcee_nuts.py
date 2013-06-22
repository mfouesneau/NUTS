""" Implements a NUTS Sampler for emcee
    http://dan.iel.fm/emcee/
"""
import numpy as np
from .nuts import nuts6, numerical_grad
from emcee.sampler import Sampler


__all__ = ['NUTSSampler', 'test_sampler']


class NUTSSampler(Sampler):
    """ A sampler object mirroring emcee.sampler object definition"""

    def __init__(self, dim, lnprobfn, gradfn=None, *args, **kwargs):
            self.dim = dim
            self.lnprobfn = _function_wrapper(lnprobfn, args)
            self.gradfn = _function_wrapper(gradfn, [])

            self.reset()

    @property
    def random_state(self):
        """
        The state of the internal random number generator. In practice, it's
        the result of calling ``get_state()`` on a
        ``numpy.random.mtrand.RandomState`` object. You can try to set this
        property but be warned that if you do this and it fails, it will do
        so silently.
        """
        pass

    @random_state.setter  # NOQA
    def random_state(self, state):
        """
        Try to set the state of the random number generator but fail silently
        if it doesn't work. Don't say I didn't warn you...

        """
        pass

    @property
    def flatlnprobability(self):
        """
        A shortcut to return the equivalent of ``lnprobability`` but aligned
        to ``flatchain`` rather than ``chain``.

        """
        return self.lnprobability.flatten()

    def get_lnprob(self, p):
        """Return the log-probability at the given position."""
        return self.lnprobfn(p, *self.args)

    def get_gradlnprob(self, p, dx=1e-3, order=1):
        """Return the log-probability at the given position."""

        if self.gradfn is not None:
            return self.gradfn(p, *self.args)
        else:
            return numerical_grad(self.lnprobfn, p, dx=dx, order=dx)

    def reset(self):
        """
        Clear ``chain``, ``lnprobability`` and the bookkeeping parameters.

        """
        self._lnprob = []
        self._chain = []
        self._epsilon = 0.

    @property
    def iterations(self):
        return len(self._lnprob)

    def clear_chain(self):
        """An alias for :func:`reset` kept for backwards compatibility."""
        return self.reset()

    def _sample_fn(self, p):
        """ proxy function for nuts6 """
        lnprob = self.lnprobfn(p)
        gradlnp = self.gradfn(p)
        return(lnprob, gradlnp)

    def sample(self, pos0, M, Madapt, delta=0.6, **kwargs):
        """ Runs NUTS6 """
        samples, lnprob, epsilon = nuts6(self._sample_fn, M, Madapt, pos0, delta)
        self._chain = samples
        self._lnprob = lnprob
        self._epsilon = epsilon

        return samples

    def run_mcmc(self, pos0, M, Madapt, delta=0.6, **kwargs):
        """
        Iterate :func:`sample` for ``N`` iterations and return the result.

        :param pos0:
            The initial position vector.

        :param M:
            The number of steps to run.

        :param Madapt:
            The number of steps to run during the burning period.

        :param delta: (optional, default=0.6)
            Initial step size.

        :param kwargs: (optional)
            Other parameters that are directly passed to :func:`sample`.

        """

        print('Running HMC with dual averaging and trajectory length %0.2f...' % delta)
        return self.sample(pos0, M, Madapt, delta, **kwargs)
        print('Done.')


class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    are also included.

    """
    def __init__(self, f, args):
        self.f = f
        self.args = args

    def __call__(self, x):
        try:
            return self.f(x, *self.args)
        except:
            import traceback
            print("NUTS: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  exception:")
            traceback.print_exc()
            raise


def test_sampler():
    """ Example usage of NUTS_sampler: sampling a 2d highly correlated Gaussian distribution """

    def correlated_normal(theta):
        """
        Example of a target distribution that could be sampled from using NUTS.
        (Although of course you could sample from it more efficiently)
        Doesn't include the normalizing constant.
        """

        # Precision matrix with covariance [1, 1.98; 1.98, 4].
        # A = np.linalg.inv( cov )
        A = np.asarray([[50.251256, -24.874372],
                        [-24.874372, 12.562814]])

        grad = -np.dot(theta, A)
        logp = 0.5 * np.dot(grad, theta.T)
        return logp, grad

    def lnprobfn(theta):
        return correlated_normal(theta)[0]

    def gradfn(theta):
        return correlated_normal(theta)[1]

    D = 2
    M = 5000
    Madapt = 5000
    theta0 = np.random.normal(0, 1, D)
    delta = 0.2

    mean = np.zeros(2)
    cov = np.asarray([[1, 1.98],
                      [1.98, 4]])

    sampler = NUTSSampler(D, lnprobfn, gradfn)
    samples = sampler.run_mcmc( theta0, M, Madapt, delta)

    print('Percentiles')
    print (np.percentile(samples, [16, 50, 84], axis=0))
    print('Mean')
    print (np.mean(samples, axis=0))
    print('Stddev')
    print (np.std(samples, axis=0))

    samples = samples[1::10, :]
    import pylab as plt
    temp = np.random.multivariate_normal(mean, cov, size=500)
    plt.plot(temp[:, 0], temp[:, 1], '.')
    plt.plot(samples[:, 0], samples[:, 1], 'r+')
    plt.show()

    return sampler
