# symengine_example

**symengine_example** is a Python application to show the ability (and the limitation) of
[symengine.py](https://github.com/symengine/symengine.py) for realistic problem.

symengine_example is an exhibition of the technology and its future outlook.
symengine_example is a learning material, not a solution for anything.

## What is symengine.py?

Faster alternative of [SymPy](https://www.sympy.org/en/index.html).
Not a perfect drop-in replacement, but has many features.

## Requirements

Python 3.11 and later.

Run `pip install -r requirements.txt`.

## How to use `visualize_spline`

For example, run:

```
python -m symengine_example.visualize_spline 2 i 2d l en
```

### The args

Before digging `visualize_spline`, let's look at M-spline and I-spline.

I-spline is a monotonic increasing spline. M-spline is non-negative spline,
which is required to make I-spline. See [Wikipedia](https://en.wikipedia.org/wiki/I-spline).

In symengine_example, I-spline is extended to n-dimensional. It makes much harder to calculate.
The complexity is a good example of realistic problem, I guess. `visualize_spline` can do 1- to 3-dimension.

M- and I-spline has the concept of "degree". More degree makes the spline smoother.
`visualize_spline` can do 1 to 3 degree.

The first three args of `visualize_spline` is, degree (1|2|3), spline type (m|i), and dimension (1d|2d|3d).

The fourth arg chooses the processing method:

1. (i) Do not use SymPy or symengine.py.
2. (s) Use `subs()` and not `lambdify()`.
3. (l) Use `lambdify()` and call the generated function.

The last arg chooses SymPy (py) or symengine.py (en). You can compare the processing time of both.

In summary:

```
python -m symengine_example.visualize_spline (1|2|3) (m|i) (1d|2d|3d) (i|s|l) (py|en)
```

### Performance comparison

The performance comparison with SymPy becomes like this:

```
$ python -m symengine_example.visualize_spline 2 i 2d l en
Total computation time: 0.8512809000094421 seconds.
    Subtotal time for function call: 0.252729299972998 seconds.
    Subtotal time for lambdify(): 0.5985516000364441 seconds.
$ python -m symengine_example.visualize_spline 2 i 2d l py
Total computation time: 83.87184929999057 seconds.
    Subtotal time for function call: 82.26858449997962 seconds.
    Subtotal time for lambdify(): 1.6032648000109475 seconds.
```

symengine.py is nearly 100 times faster even including `lambdify()` time!
If excluding, 300x! (symengine-0.10.0)

IMPORTANT: **The result of `lambdify()` is pickle-able.**

### Memory usage observation (and AMD CPU vulnerability?)

Open Task Manager (if Windows), select Performance -> Memory, and run
`python -m symengine_example.visualize_spline 2 i 3d l en`.

You can see 400MB of increasing in Committed while computing (symengine-0.10.0).

So next, close all other apps, and run `python -m symengine_example.visualize_spline 3 i 3d l en`.
it exhausts Committed 30 GB or more, and aborts.

On one of my PCs, it crashes whole Windows (symengine-0.10.0).
I guess it exploits a vulnerability of AMD CPU (Ryzen 5 5600X).

## How to use `test_model.py`

Before digging `test_model.py`, let's look at n-dimensional I-spline a bit more.

We introduce the expression `y = f(x)` as n-dimensional I-spline.
`x` is an n-dimensional vector, and `y` is a scalar.

`visualize_spline` shows you the form of n-dimensional I-spline, and you can see that
`y` is always 0 when `any(x == 0)`. This is a critical limitation for spline.
So I made a model which doesn't have the problem by combining I-spline.
The detail is in `model.py`. For now, leave it as "the model".

Run `python -m unittest tests.test_model.TestModel.test_fit_params_solve_xs`
and wait one or several minutes. It fits a 3-dimensional vector field to the model.
You can see the result by `python -m tests.test_model`.

`test_model.py` demonstrates the outlook of symengine.py: **The integration of
numerical analysis and computer algebra with practical performance for realistic problem!**
Moreover, currently `test_model.py` doesn't use `lambdify()` for the memory usage problem mentioned above.
If the problem has been fixed, **it will be a revolution of computing!**

## License

MIT license.
