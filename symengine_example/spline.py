from itertools import product
from functools import reduce
import numpy as np
from symengine_example.sym import Symbol, Piecewise, And, Or, LessThan, StrictLessThan, Ne, Mul, Eq


class _Array:
    '''
    Substitute of sympy.Array for symengine.

    sympy.Array can return formula in __getitem__() but symengine doesn't have such class.

    Parameters
    ----------
    values : Iterable of (float | Symbol).
    '''
    def __init__(self, values) -> None:
        values = np.asarray(values)
        ndim = len(values.shape)
        self.s_indices = [Symbol('i' + str(i), integer=True) for i in range(ndim)]
        self.pw = Piecewise(*self._build_dim(values, ndim))
        self.values = values
    
    def _build_dim(self, values, dim):
        pw = []
        for i, v in enumerate(values):
            eq = Eq(i, self.s_indices[dim - 1])
            if dim == 1:
                pw.append((v, eq))
            else:
                pw.append((Piecewise(*self._build_dim(v, dim - 1)), eq))
        pw.append((0, True))
        return pw

    def __getitem__(self, indices):
        sub = {}
        for i, si in zip(indices, self.s_indices[::-1]):
            sub[si] = i
        return self.pw.subs(sub)

    def __iter__(self):
        return iter(self.values)

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim


class NdimLinearInterpolator:
    '''
    symengine version of scipy.interpolate.RegularGridInterpolator.
    __call__ can return a formula of symengine.
    Unlike scipy.interpolate.RegularGridInterpolator, __call__ cannot accept a list of coordinates.

    Parameters
    ----------
    grid : Iterable of Iterable of (float | Symbol). Numpy value is not acceptable.
    values : Iterable of (float | Symbol). Numpy value is not acceptable.
        Data values at the grid.

    Examples
    --------
    >>> import numpy as np
    >>> s_x = Symbol('x')
    >>> xz = [float(v) for v in np.arange(2)]
    >>> y = [float(v) for v in np.arange(3)]
    >>> data = np.array([
    ...     [[0, 0.2], [0.3, 0.5], [s_x, 0.6]],
    ...     [[0.6, 0.8], [0.4, 1], [0.5, 1.1]],
    ... ])
    >>> interp = NdimLinearInterpolator((xz, y, xz), data)
    >>> print(interp((0.5, 1.2, 0.5)))
    Should print ``0.55 + 0.05*x``.

    '''
    def __init__(self, grid, values) -> None:
        self.grid = grid
        self.values = _Array(values)

        grid_indices = []
        grid_distances = []
        grid_v = []
        for ii, g in enumerate(grid):
            v = Symbol('v' + str(ii), real=True, positive=True)
            ipt = []
            dpt = []
            for i, (g0, g1) in enumerate(zip(g[:-1], g[1:])):
                ipt.append((i, And(g0 <= v, v < g1)))
                dpt.append(((v - g0) / (g1 - g0), And(g0 <= v, v < g1)))
            ipt.append((0, True))
            dpt.append((0, True))
            grid_indices.append(Piecewise(*ipt))
            grid_distances.append(Piecewise(*dpt))
            grid_v.append(v)
        self.grid_indices = grid_indices
        self.grid_distances = grid_distances
        self.grid_v = grid_v

    def __call__(self, xi):
        '''
        Interpolates.

        Parameters
        ----------
        xi : tuple of float or NDArray[float]. Should be len(xi) == len(grid).
        '''
        xi = np.asarray(xi)
        if len(self.grid) == 1:
            # if xi.shape != ():
            #     raise Exception('Bad argument.')
            xi = xi.reshape(1)
        elif xi.shape[0] != len(self.grid):
            raise Exception('Bad argument.')
        index, norm_distance = self._find_index(xi)
        return self._evaluate_linear(index, norm_distance)

    def _find_index(self, xi):
        index, norm_distance = np.empty_like(xi, object), np.empty_like(xi, object)
        for i in range(xi.shape[0]):
            v = xi[i]
            gv = self.grid_v[i]
            index[i] = self.grid_indices[i].subs({gv: v})
            norm_distance[i] = self.grid_distances[i].subs({gv: v})
        return index, norm_distance

    def _evaluate_linear(self, index, norm_distance):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,) * (self.values.ndim - len(index))

        # Compute shifting up front before zipping everything together
        shift_norm_distance = 1 - norm_distance
        shift_index = index + 1

        # The formula for linear interpolation in 2d takes the form:
        # values = self.values[(i0, i1)] * (1 - y0) * (1 - y1) + \
        #          self.values[(i0, i1 + 1)] * (1 - y0) * y1 + \
        #          self.values[(i0 + 1, i1)] * y0 * (1 - y1) + \
        #          self.values[(i0 + 1, i1 + 1)] * y0 * y1
        # We pair i with 1 - yi (zipped1) and i + 1 with yi (zipped2)
        zipped1 = zip(index, shift_norm_distance)
        zipped2 = zip(shift_index, norm_distance)

        # Take all products of zipped1 and zipped2 and iterate over them
        # to get the terms in the above formula. This corresponds to iterating
        # over the vertices of a hypercube.
        hypercube = product(*zip(zipped1, zipped2))
        value = np.array([0.])
        for h in hypercube:
            edge_indices, weights = zip(*h)
            weight = np.array([1.])
            for w in weights:
                weight = weight * w
            term = np.asarray(self.values[edge_indices[::-1]]) * weight[vslice]
            value = value + term   # cannot use += because broadcasting
        return value[0]


def _all(f, x, y):
    return And(*map(lambda a: f(*a), [o for o in zip(x, y)]))


def _any(f, x, y):
    return Or(*map(lambda a: f(*a), [o for o in zip(x, y)]))


def spline_i_m(DEGREE: int, N_KNOT: int, N_DIM: int, X_MIN, X_MAX, S_KNOTS=None):
    '''
    Constructs symengine Symbol as S_KNOTS. function_m() and function_i() constructs
    symengine value depends on x, namely, function.

    Parameters
    ----------
    DEGREE : int
        Degree of a spline.
    N_KNOT : int
        Number of knot of a spline.
    N_DIM : int
        Dimension of a spline.
    X_MIN : float | Symbol
        Min of x.
    X_MAX : float | Symbol
        Max of x.
    S_KNOTS : NDArray[object], the shape should be (N_KNOT, N_DIM), optional
        The NDArray type argument should be object!

    Returns
    --------
    S_KNOTS : [[Symbol] * N_KNOW] * N_DIM
    function_m : Callable[[degree: int, idx_spline: [int] * N_DIM, x: [float | Symbol] * N_DIM], float]
        M-spline function.
    function_i : Callable[[idx_spline: [int] * N_DIM, x: [float | Symbol] * N_DIM], float]
        I-spline function.
    '''

    if S_KNOTS is None:
        S_KNOTS = np.empty((N_KNOT, N_DIM), object)
        for i in product(range(N_KNOT), range(N_DIM)):
            S_KNOTS[i] = Symbol('T' + str(i), real=True, positive=True)
    TS = np.empty((DEGREE * 2 + N_KNOT + 1,) * N_DIM + (N_DIM,), object)
    _ts = [None] * N_DIM
    for d in range(N_DIM):
        sum_s_knots = sum(S_KNOTS[:, d])
        align_s_knots = [
            sum([
                S_KNOTS[ii, d]
                for ii in range(i + 1)
            ]) / (sum_s_knots + 1.)
            for i in range(N_KNOT)
        ]
        _ts[d] = [X_MIN] * DEGREE + [
            (X_MAX - X_MIN) * t + X_MIN
            for t in align_s_knots
        ] + [X_MAX] * (DEGREE + 1)
    for i in product(*[range(DEGREE * 2 + N_KNOT + 1)] * N_DIM):
        TS[i] = tuple(_ts[d][i[d]] for d in range(N_DIM))
    del _ts

    def _func_m(degree, idx_spline, x):
        grid = [
            [TS[tuple(idx_spline)][d], TS[tuple(idx_spline + degree)][d]]
            for d in range(N_DIM)
        ]
        values = [
            function_m(degree - 1, idx_spline + i[::-1], x)
            for i in product(*[[1, 0]] * N_DIM)
        ]
        values = np.array(values).reshape((2,) * N_DIM).tolist()
        interp = NdimLinearInterpolator(grid, values)
        return degree * interp(x) / (degree - 1)

    def function_m(degree, idx_spline, x):
        if degree == 1:
            rec_y = Piecewise(
                (reduce(Mul, TS[tuple(idx_spline + 1)] - TS[tuple(idx_spline)], 1),
                 _all(StrictLessThan, TS[tuple(idx_spline)], TS[tuple(idx_spline + 1)])),
                (1., True)
            )
            y = Piecewise(
                (1. / rec_y, And(
                    _all(LessThan, TS[tuple(idx_spline)], x),
                    _all(StrictLessThan, x, TS[tuple(idx_spline + 1)]))),
                (0., True)
            )
        else:
            y = Piecewise(
                (_func_m(degree, idx_spline, x),
                 _all(Ne, TS[tuple(idx_spline + degree)], TS[tuple(idx_spline)])),
                (0., True)
            )
        return y

    def _f_a(idx_spline, x):
        if np.any(DEGREE * 2 + N_KNOT + 1 <= idx_spline + DEGREE + 1):
            return 0.
        _a = DEGREE + 1
        _b = reduce(Mul, TS[tuple(idx_spline + DEGREE + 1)] - TS[tuple(idx_spline)], 1)
        _c = function_m(DEGREE + 1, idx_spline, x)
        return _b * _c / _a

    def function_i(idx_spline, x):
        j = [
            Piecewise(*([
                (i[d], And(
                    _all(LessThan, TS[i], x),
                    _all(StrictLessThan, x, TS[tuple(np.array(i) + 1)])
                ))
                for i in product(*[range(DEGREE * 2 + N_KNOT)] * N_DIM)
            ] + [(0, True)]))
            for d in range(N_DIM)
        ]
        return sum(
            Piecewise(
                (_f_a(np.array(_m), x), _any(LessThan, _m, j)),
                (0., True)
            )
            for _m in product(*[
                range(idx_spline[d], DEGREE + N_KNOT)
                for d in range(N_DIM)
            ])
        )

    return S_KNOTS, function_m, function_i
