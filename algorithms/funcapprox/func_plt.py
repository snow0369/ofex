from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def plot_functions(func_list: Dict[str, sp.Expr],
                   x: sp.Symbol,
                   x_points: Optional[Sequence[float]] = None,
                   title: Optional[str] = None,):
    if x_points is None:
        x_points = np.linspace(-1, 1, 100)
    plt_imag = False
    y_points = dict()
    for func_name, func in func_list.items():
        y_points[func_name] = sp.lambdify(x, func, "numpy")(x_points)
        if not np.allclose(y_points[func_name].imag, 0.0):
            plt_imag = True

    fig = plt.figure()
    if not plt_imag:
        ax = fig.add_subplot(111)
        for func_name, func in func_list.items():
            ax.plot(x_points, y_points[func_name].real, label=func_name)
        ax.legend()
    else:
        axes = fig.subplots(2)
        for func_name, func in func_list.items():
            axes[0].plot(x_points, y_points[func_name].real, label=func_name)
            axes[1].plot(x_points, y_points[func_name].imag, label=func_name)
        axes[1].legend()
    if title is not None:
        plt.title(title)
    plt.show()
