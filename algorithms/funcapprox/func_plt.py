from typing import List, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def plot_functions(func_list: Dict[str, sp.Expr],
                   x: sp.Symbol,
                   x_points: Optional[Sequence[float]] = None):
    if x_points is None:
        x_points = np.linspace(-1, 1, 100)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for func_name, func in func_list.items():
        y_points = sp.lambdify(x, func, "numpy")(x_points)
        ax.plot(x_points, y_points, label=func_name)
    ax.legend()
    plt.show()
