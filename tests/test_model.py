import faulthandler
faulthandler.enable()
import unittest
from itertools import product
import pickle
import numpy as np
from symengine_example import enable_symengine
enable_symengine()
from symengine_example import model, CURRENT_DIR


DEGREE = 2
N_KNOT = 1
N_DIM = 3
X_MIN = 0.
X_MAX = 1.

TMP = CURRENT_DIR / 'tmp'
PKL = TMP / 'test_fit_params_solve_xs.pkl'


def generate_ys(XS, indices):
    '''
    Monotonic increasing function.
    '''
    return np.array(list(map(
        lambda x: sum(np.sin(_x) / (i + 1) for i, _x in enumerate(x[indices])), XS
    )))


N_XS = 6  # Large enough value to determine parameters.
XS = np.array(list(product(*[np.linspace(X_MIN, X_MAX - 1e-10, N_XS)] * N_DIM)))
YS = np.array([generate_ys(XS, [0, 1, 2]), generate_ys(XS, [1, 2, 0]), generate_ys(XS, [2, 0, 1])]).transpose(1, 0)
KNOTS = np.full((N_KNOT, N_DIM), 1., object)  # knots can be optimized too, but it consumes too much time now (symengine-0.10.0).


class TestModel(unittest.TestCase):
    def test_construct_model_formula(self):
        '''
        Just constructs a model formula and checks its basic properties.
        '''
        s_y, s_x, s_weight, s_knot = model.construct_model_formula(DEGREE, N_KNOT, N_DIM, X_MIN, X_MAX)
        self.assertEqual(len(s_weight), 64)
        self.assertEqual(len(s_knot), 3)
        a = s_y.subs({
            s: 0.1 for s in s_weight + s_knot
        }).subs({
            s: 0.1 for s in s_x
        })
        float(a)  # assert it can be float, not an expression

    def test_fit_params_solve_xs(self):
        '''
        1) Fits source data to the model by optimizing the parameters.
        2) With the fitted model, solves x from y.
        This test consumes one or more minutes. Can resume after fit_params().
        '''
        s_y, s_x, s_weight, _ = model.construct_model_formula(DEGREE, N_KNOT, N_DIM, X_MIN, X_MAX, KNOTS)

        if not TMP.exists():
            TMP.mkdir()
        if not PKL.exists():
            fitted_params = model.fit_params(XS, YS, np.ones_like(YS), s_y, s_x, s_weight, [])

            with open(PKL, 'wb') as f:
                pickle.dump(fitted_params, f)
        else:
            with open(PKL, 'rb') as f:
                fitted_params = pickle.load(f)

        func = model.generate_func(fitted_params, s_y, s_x, s_weight, [])
        for _x, _y in zip(XS, YS):
            for _s, _e in zip(_y, func(*_x)):
                self.assertAlmostEqual(_s, _e, delta=0.002)

        ys_to_solve = np.array([
            YS[22],  # should be solved
            [-0.1] * N_DIM  # should fail
        ])
        solved_ys, solved_xs, failed_ys = model.solve_xs(
            func, ys_to_solve,
            XS[21],  # starts from neighbor
            N_DIM, X_MIN, X_MAX)
        self.assertEqual(len(solved_ys), 1)  # YS[22]
        self.assertEqual(len(solved_xs), 1)  # XS[22]
        self.assertEqual(len(failed_ys), 1)  # [-0.1] * N_DIM
        for _s, _e in zip(XS[22], solved_xs[0]):
            self.assertAlmostEqual(_s, _e, delta=1e-3)


def main():
    '''
    Visualizes the source data and the fitted model.
    '''
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if not TMP.exists() or not PKL.exists():
        raise Exception('Run "python -m unittest tests.test_model.TestModel.test_fit_params_solve_xs" first.')
    with open(PKL, 'rb') as f:
        fitted_params = pickle.load(f)
    s_y, s_x, s_weight, _ = model.construct_model_formula(DEGREE, N_KNOT, N_DIM, X_MIN, X_MAX, KNOTS)
    func = model.generate_func(fitted_params, s_y, s_x, s_weight, [])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(X_MIN, X_MAX)
    ax.set_zlim(X_MIN, X_MAX)
    fit_ys = np.array([func(*x) for x in XS])
    p0 = ax.quiver(XS[:, 0], XS[:, 1], XS[:, 2],
                   YS[:, 0], YS[:, 1], YS[:, 2],
                   length=0.1, colors='blue')
    p1 = ax.quiver(XS[:, 0], XS[:, 1], XS[:, 2],
                   fit_ys[:, 0], fit_ys[:, 1], fit_ys[:, 2],
                   length=0.1, colors='red')
    animation.ArtistAnimation(fig, [[p0], [p1]], interval=500)
    plt.title('blue: source data     red: fitted model')
    plt.show()


if __name__ == '__main__':
    main()
