""" Some helpers to help the usage of NUTS
This package contains:

  numerical_grad            return numerical estimate of the local gradient
_function_wrapper           hack to make partial functions pickleable
NutsSampler_fn_wrapper      combine provided lnp and grad(lnp) into one function
"""
import numpy as np


def numerical_grad(theta, f, dx=1e-3, order=1):
    """ return numerical estimate of the local gradient

    The gradient is computer by using the Taylor expansion approximation over
    each dimension:
        f(t + dt) = f(t) + h df/dt(t) + h^2/2 d^2f/dt^2 + ...

    The first order gives then:
        df/dt = (f(t + dt) - f(t)) / dt + O(dt)
    Note that we could also compute the backwards version by subtracting dt instead:
        df/dt = (f(t) - f(t -dt)) / dt + O(dt)

    A better approach is to use a 3-step formula where we evaluate the
    derivative on both sides of a chosen point t using the above forward and
    backward two-step formulae and taking the average afterward. We need to use the Taylor expansion to higher order:
        f (t +/- dt) = f (t) +/- dt df/dt + dt ^ 2 / 2  dt^2 f/dt^2 +/- dt ^ 3 d^3 f/dt^3 + O(dt ^ 4)

        df/dt = (f(t + dt) - f(t - dt)) / (2 * dt) + O(dt ^ 3)

    Note: that the error is now of the order of dt ^ 3 instead of dt

    In a same manner we can obtain the next order by using f(t +/- 2 * dt):
        df/dt = (f(t - 2 * dt) - 8 f(t - dt)) + 8 f(t + dt) - f(t + 2 * dt) / (12 * dt) + O(dt ^ 4)

    In the 2nd order, two additional function evaluations are required (per dimension), implying a
    more time-consuming algorithm. However the approximation error is of the order of dt ^ 4


    INPUTS
    ------
    theta: ndarray[float, ndim=1]
        vector value around which estimating the gradient
    f: callable
        function from which estimating the gradient

    KEYWORDS
    --------
    dx: float
        pertubation to apply in each direction during the gradient estimation
    order: int in [1, 2]
        order of the estimates:
            1 uses the central average over 2 points
            2 uses the central average over 4 points

    OUTPUTS
    -------
    df: ndarray[float, ndim=1]
        gradient vector estimated at theta

    COST: the gradient estimation need to evaluates ndim * (2 * order) points (see above)
    CAUTION: if dt is very small, the limited numerical precision can result in big errors.
    """
    ndim = len(theta)
    df = np.empty(ndim, dtype=float)
    if order == 1:
        cst = 0.5 / dx
        for k in range(ndim):
            dt = np.zeros(ndim, dtype=float)
            dt[k] = dx
            df[k] = (f(theta + dt) - f(theta - dt)) * cst
    elif order == 2:
        cst = 1. / (12. * dx)
        for k in range(ndim):
            dt = np.zeros(ndim, dtype=float)
            dt[k] = dx
            df[k] = cst * (f(theta - 2 * dt) - 8. * f(theta - dt) + 8. * f(theta + dt) - f(theta + 2. * dt) )
    return df


class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    are also included.
    """
    def __init__(self, f, args=(), kwargs={}):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:
            import traceback
            print("NUTS: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


class NutsSampler_fn_wrapper(object):
    """ Create a function-like object that combines provided lnp and grad(lnp)
    functions into one as required by nuts6.

    Both functions are stored as partial function allowing to fix arguments if
    the gradient function is not provided, numerical gradient will be computed

    By default, arguments are assumed identical for the gradient and the
    likelihood. However you can change this behavior using set_xxxx_args.
    (keywords are also supported)

    if verbose property is set, each call will print the log-likelihood value
    and the theta point
    """
    def __init__(self, lnp_func, gradlnp_func=None, *args, **kwargs):
        self.lnp_func = _function_wrapper(lnp_func, args, kwargs)
        if gradlnp_func is not None:
            self.gradlnp_func = _function_wrapper(gradlnp_func, args, kwargs)
        else:
            self.gradlnp_func = _function_wrapper(numerical_grad, (self.lnp_func,))
        self.verbose = False

    def set_lnp_args(self, *args, **kwargs):
            self.lnp_func.args = args
            self.lnp_func.kwargs = kwargs

    def set_gradlnp_args(self, *args, **kwargs):
            self.gradlnp_func.args = args
            self.gradlnp_func.kwargs = kwargs

    def __call__(self, theta):
        r = (self.lnp_func(theta), self.gradlnp_func(theta))
        if self.verbose:
            print(r[0], theta)
        return r
