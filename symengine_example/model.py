import typing as ty
from itertools import product, combinations
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit, least_squares
from symengine_example.sym import Symbol, Matrix, Basic
from symengine_example.spline import spline_i_m


def construct_model_formula(DEGREE: int, N_KNOT: int, N_DIM: int, X_MIN: float, X_MAX: float, S_KNOTS=None) \
        -> tuple[Basic, NDArray, list[Symbol], list[float | Symbol]]:
    '''
    Constructs a model formula.
    Model: ``f(x0, x1, ... xn) -> y``
    All variables are scalar. ``x`` is bound to [X_MIN, X_MAX).
    ``f()`` is determined by ``s_weight`` and ``s_knot``.
    You should fit ``s_weight`` and ``s_knot`` to measured results.

    The model consists of four elements.
    1) N_DIM, (N_DIM - 1), ...1-dimensional I-spline
    2) Weights of each spline
    3) Origin
    Simple multiply-accumulation (1 * 2 + 3) is the model.
    ``s_knot`` determines 1. 2 and 3 are ``s_weight``.

    Parameters
    ----------
    S_KNOTS : NDArray of (float | Symbol). The shape should be ``(N_KNOT, N_DIM)``.

    Returns
    --------
    s_y : Expression of ``y``.
    s_x : NDArray of Symbol of ``x``. The shape is ``(N_DIM,)``.
    s_weight : list[Symbol] of parameter.
    s_knot : list of (float | Symbol). Should be positive. The length is ``N_KNOT * N_DIM``.
    '''

    if S_KNOTS is None:
        S_KNOTS = np.empty((N_KNOT, N_DIM), object)
        for i in product(range(N_KNOT), range(N_DIM)):
            S_KNOTS[i] = Symbol('T' + str(i), real=True, positive=True)

    function_i_nd = []
    for d in range(N_DIM):
        function_i_d = []
        for _ii in combinations(range(N_DIM), (d + 1)):
            _, _, function_i = spline_i_m(DEGREE, N_KNOT, (d + 1), X_MIN, X_MAX, S_KNOTS[:, list(_ii)])
            function_i_d.append(function_i)
        function_i_nd.append(function_i_d)

    s_x = np.array([Symbol('x' + str(d), real=True) for d in range(N_DIM)])
    s_ys = []
    s_ws = []
    for d in range(N_DIM):
        function_i_d = function_i_nd[d]
        for _i in product(*([range(DEGREE + N_KNOT)] * (d + 1))):
            idx_spline = np.array(_i)
            for _ii, function_i in zip(combinations(range(N_DIM), (d + 1)), function_i_d):
                s_w = Symbol(f'w_{d + 1}_{_i}_{_ii}', real=True)
                s_ws.append(s_w)
                s_ys.append(function_i(idx_spline, s_x[list(_ii)]) * s_w)

    s_origin = Symbol('o', real=True)
    s_y = sum(s_ys) + s_origin
    return s_y, s_x, s_ws + [s_origin], S_KNOTS.flatten().tolist()


def fit_params(xs: NDArray[np.floating], ys: NDArray[np.floating], sigma: NDArray[np.floating], s_y, s_x, s_weight, s_knot) \
        -> list[list[float]]:
    '''
    Receives measured results and returns tone reproduction curves as functions.

    By using lambdify(), curve_fit() can be faster 100 times, but it consumes too much memory
    and crashes OS now (symengine-0.10.0).

    Parameters
    ----------
    xs : NDArray[np.floating]
        Source of measured result. The shape should be (N, N_DIM).
    ys : NDArray[np.floating]
        Measured result itself. The shape should be (N, n_dim_y).
    sigma : NDArray[np.floating]
        Sigma for curve fitting. The shape should be (N, n_dim_y).

    Returns
    --------
    fitted_params : list[list[float] * len(s_weight + s_knot)] * n_dim_y
    '''
    for arr in [xs, ys, sigma]:
        if arr.ndim != 2:
            raise Exception()
    if sigma.shape != ys.shape:
        raise Exception()
    if ys.shape[0] != xs.shape[0]:
        raise Exception()

    s_y_xs = Matrix([
        s_y.subs({
            _s_x: _x
            for _s_x, _x in zip(s_x, x)
        })
        for x in xs
    ])

    def func(_, *param):
        _ys = s_y_xs.subs({
            _s: _p
            for _s, _p in zip(s_weight + s_knot, param)
        })
        return list(map(float, _ys))

    s_jac_y_xs = s_y_xs.jacobian(Matrix(s_weight + s_knot))

    def jac_func(_, *params):
        _ys = s_jac_y_xs.subs({
            _s: _p
            for _s, _p in zip(s_weight + s_knot, params)
        })
        return np.array(list(map(float, _ys))).reshape(s_jac_y_xs.shape)

    wb = (-np.inf, np.inf)  # Usually I-spline takes non-negative weight for monotonicity.
    kb = (1e-10, np.inf)  # In reality, knot is not used in test_model.py. It makes quite slower.
    fitted_params = []
    for d in range(ys.shape[1]):
        fp, pcov = curve_fit(
            func,
            xs,
            ys[:, d],
            (0.,) * len(s_weight) + (1.,) * len(s_knot),
            bounds=np.array((wb,) * len(s_weight) + (kb,) * len(s_knot)).transpose(1, 0),
            sigma=sigma[:, d],
            jac=jac_func
        )
        fitted_params.append(fp)
    
    return fitted_params


def generate_func(params, s_y, s_x, s_weight, s_knot):
    '''
    Generates functions of ``params``.

    Returns
    --------
    func : Callable[list[float] * N_DIM, [float] * len(params)]
    '''
    fs = []

    def f0(s_y_param):
        def f(*x):
            return float(s_y_param.subs({_s: _x for _s, _x in zip(s_x, x)}))
        return f

    for p in params:
        s_y_param = s_y.subs({
            _s_p: _p
            for _s_p, _p in zip(s_weight + s_knot, p)
        })
        fs.append(f0(s_y_param))

    def func(*x):
        return [f(*x) for f in fs]

    return func


def _np_array_reshape(vs, n_dim):
    return np.array(vs).reshape(-1, n_dim)


def solve_xs(func: ty.Callable, ys: NDArray[np.floating], first_x0: NDArray[np.floating], N_DIM: int, X_MIN: float, X_MAX: float):
    '''
    Solves N_DIM length array x in ``y = f(x)`` for f in fs, for y in ys.
    It searches solution from initial guess value ``x0``. So it requires first ``x0`` for the first search on ys[0].
    That is ``first_x0``. After first iteration, it searches the nearest solved y and uses the solution.

    Parameters
    ----------
    func : ty.Callable
        The length of return value should be ``n_dim_y``. ty.Callable args should be list[float] * N_DIM and returns float.
    ys : NDArray[np.floating]
        The shape should be ``(N, n_dim_y)``.
    first_x0 : NDArray[np.floating]
        Initial guess value ``x0`` of ``ys[0]``. The shape should be ``(N_DIM)``.

    Returns
    --------
    solved_ys : NDArray[np.floating]
        list[y] which has solution. The shape is ``(n_solved, n_dim_y)``.
    solved_xs : NDArray[np.floating]
        Solutions. The shape is ``(n_solved, N_DIM)``.
    failed_ys : NDArray[np.floating]
        list[y] which was failed to solve. The shape is ``(N - n_solved, n_dim_y)``.
    '''
    if ys.ndim != 2 or first_x0.shape != (N_DIM,):
        raise Exception()
    n_dim_y = ys.shape[1]

    def minimize_func(x, *y):
        return [_e - _y for _e, _y in zip(func(*x), y)]

    solved_xs = []
    solved_ys = []
    failed_ys = []
    x0 = first_x0
    for y in ys:
        if len(solved_ys) > 0:
            _solved = np.array(solved_ys)
            i = np.argmin(np.linalg.norm(_solved - y, axis=1))
            x0 = solved_xs[i]
        res = least_squares(minimize_func, x0, bounds=(X_MIN, X_MAX - 1e-10), args=tuple(y))
        if res.cost < 1e-5:
            solved_xs.append(res.x)
            solved_ys.append(y)
        else:
            failed_ys.append(y)

    return _np_array_reshape(solved_ys, n_dim_y), _np_array_reshape(solved_xs, N_DIM), _np_array_reshape(failed_ys, n_dim_y)
