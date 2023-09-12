import faulthandler
faulthandler.enable()
from itertools import product
import sys
from enum import Enum, auto
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


N_KNOT = 1
X_MIN = 0.
X_MAX = 1.

TOTAL_LAMB = 0.
TOTAL_CALL = 0.


class Method(Enum):
    Immediate = auto()
    Subs = auto()
    Lambdify = auto()


def compute(method: Method, func, i):
    global TOTAL_LAMB, TOTAL_CALL
    idx_spline = np.array(i)
    if method == Method.Immediate:
        before = perf_counter()
        ys = [func(idx_spline, x) for x in XS]
        after = perf_counter()
    else:
        s_y = func(idx_spline, S_X)
        if method == Method.Subs:
            before = perf_counter()
            ys = [float(s_y.subs({S_X[d]: x[d] for d in range(N_DIM)})) for x in XS]
            after = perf_counter()
        elif method == Method.Lambdify:
            lamb = perf_counter()
            f = lambdify(S_X, [s_y])
            before = perf_counter()
            ys = [f(*x) for x in XS]
            after = perf_counter()
            TOTAL_LAMB += before - lamb
        else:
            raise Exception()
        TOTAL_CALL += after - before
    return ys


def print_processing_time():
    print(f'Total computation time: {TOTAL_LAMB + TOTAL_CALL} seconds.')
    if TOTAL_LAMB > 0.:
        print(f'    Subtotal time for function call: {TOTAL_CALL} seconds.')
        print(f'    Subtotal time for lambdify(): {TOTAL_LAMB} seconds.')


def main_1d(method: Method, func, func_name: str):
    for i in range(DEGREE + N_KNOT):
        ys = compute(method, func, i)
        plt.plot(XS, ys, label=str(i))
    print_processing_time()
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.title(f'1-D {func_name}')
    plt.show()


def main_2d(method: Method, func, func_name: str):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(X_MIN, X_MAX)
    ims = []
    for _i in product(*([range(DEGREE + N_KNOT)] * N_DIM)):
        ys = compute(method, func, _i)
        px0, px1 = np.array(XS).reshape(N_X, N_X, 2).transpose(2, 0, 1)
        p = ax.plot_surface(px0, px1, np.array(ys).reshape(N_X, N_X))
        ims.append([p])
    print_processing_time()
    _ = animation.ArtistAnimation(fig, ims, interval=500)
    plt.title(f'2-D {func_name}')
    plt.show()


def main_3d(method: Method, func, func_name: str):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(X_MIN, X_MAX)
    ax.set_zlim(X_MIN, X_MAX)
    ims = []
    for ii, i in enumerate(product(*([range(DEGREE + N_KNOT)] * N_DIM))):
        ys = compute(method, func, i)
        visibility_param = 10 if func_name == 'I-spline' else 1
        p = ax.scatter([x[0] for x in XS], [x[1] for x in XS], [x[2] for x in XS], marker='o',
                       s=np.array(ys) * visibility_param,
                       c=f'C{ii}')
        ims.append([p])
    print_processing_time()
    _ = animation.ArtistAnimation(fig, ims, interval=500)
    plt.title(f'3-D {func_name}')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Four arguments are required. (1|2|3) (m|i) (1d|2d|3d) (s|l|i) (py|en)')
        sys.exit(1)

    from symengine_example import enable_symengine
    if sys.argv[5] == 'py':
        pass
    elif sys.argv[5] == 'en':
        enable_symengine()
    else:
        print('Error: Fifth argument should be py, or en.')
        sys.exit(1)
    from symengine_example.sym import Symbol, lambdify
    from symengine_example.spline import spline_i_m

    if sys.argv[1] == '1':
        DEGREE = 1
    elif sys.argv[1] == '2':
        DEGREE = 2
    elif sys.argv[1] == '3':
        DEGREE = 3
    else:
        print('Error: First argument should be 1, 2, or 3.')
        sys.exit(1)

    if sys.argv[4] == 's':
        method = Method.Subs
    elif sys.argv[4] == 'l':
        method = Method.Lambdify
    elif sys.argv[4] == 'i':
        method = Method.Immediate
    else:
        print('Error: Second argument should be s, l, or i.')
        sys.exit(1)

    if sys.argv[3] == '1d':
        N_DIM = 1
        main = main_1d
    elif sys.argv[3] == '2d':
        N_DIM = 2
        main = main_2d
    elif sys.argv[3] == '3d':
        N_DIM = 3
        main = main_3d
    else:
        print('Error: Fourth argument should be 1d or 3d.')
        sys.exit(1)

    N_X = 10 if N_DIM == 3 else 100
    XS = list(product(*[np.linspace(X_MIN, X_MAX - 1e-5, N_X)] * N_DIM))
    S_X = np.array([Symbol('x' + str(d), real=True, positive=True) for d in range(N_DIM)])

    knots = np.full((N_KNOT, N_DIM), 1., object)
    _, function_m, function_i = spline_i_m(DEGREE, N_KNOT, N_DIM, X_MIN, X_MAX, knots)

    if sys.argv[2] == 'm':
        def func(idx_spline, x):
            return function_m(DEGREE, idx_spline, x)
        func_name = 'M-spline'
    elif sys.argv[2] == 'i':
        def func(idx_spline, x):
            return function_i(idx_spline, x)
        func_name = 'I-spline'
    else:
        print('Error: Second argument should be m, or i.')
        sys.exit(1)

    main(method, func, func_name)
