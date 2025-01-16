from typing import Dict, Optional, Sequence, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

__all__ = ["plot_functions"]


def plot_functions(func_list: Dict[str, sp.Expr],
                   x: sp.Symbol,
                   plot_options: Optional[Dict[str, Dict[str, Any]]] = None,
                   x_points: Optional[Sequence[float]] = None,
                   title: Optional[str] = None,):
    """
    Plots one or more mathematical functions defined as SymPy expressions.

    Args:
        func_list (Dict[str, sp.Expr]): A dictionary where keys are function names (str)
                                         and values are SymPy expressions representing the functions to be plotted.
        x (sp.Symbol): The variable (symbol) used in the expressions within `func_list`.
        plot_options (Optional[Dict[str, Dict[str, Any]]]): A nested dictionary where the outer keys correspond
                                                            to the function names in `func_list` and the inner keys
                                                            specify matplotlib keyword arguments controlling the
                                                            appearance of the individual plots (e.g., color, linestyle).
        x_points (Optional[Sequence[float]]): A sequence of x-values where the function(s) will be evaluated.
                                              If None, it defaults to 100 evenly spaced points between -1 and 1.
        title (Optional[str]): A title for the plot. If None, no title is added.

    Returns:
        None: Displays the plot(s) using matplotlib.
    """
    
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
            if plot_options is not None and func_name in plot_options:
                kwargs = plot_options[func_name]
            else:
                kwargs = dict()
            ax.plot(x_points, y_points[func_name].real, label=func_name, **kwargs)
        ax.legend()
    else:
        axes = fig.subplots(2)
        for func_name, func in func_list.items():
            if plot_options is not None and func_name in plot_options:
                kwargs = plot_options[func_name]
            else:
                kwargs = dict()
            axes[0].plot(x_points, y_points[func_name].real, label=func_name, **kwargs)
            axes[1].plot(x_points, y_points[func_name].imag, label=func_name, **kwargs)
        axes[1].legend()
    if title is not None:
        plt.title(title)
    plt.show()
