from typing import Dict, Optional, Sequence, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

__all__ = ["plot_functions"]


def plot_functions(func_list: Dict[str, sp.Expr],
                   x: sp.Symbol,
                   plot_options: Optional[Dict[str, Dict[str, Any]]] = None,
                   range_plot: Optional[Dict[str, Tuple[float, float]]] = None,
                   range_plot_options: Optional[Dict[str, Dict[str, Any]]] = None,
                   x_points: Optional[Sequence[float]] = None,
                   title: Optional[str] = None,
                   plot: bool = True,
                   axes: Optional[plt.Axes] = None):
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
        range_plot (Optional[Dict[str, Tuple[float, float]]]): A dictionary where the keys are function names
                                                               and the values are tuples specifying the minimum and
                                                               maximum vertical distances to highlight as a range for a function.
        range_plot_options (Optional[Dict[str, Dict[str, Any]]]): A nested dictionary where outer keys correspond
                                                                   to the function names in `range_plot`, and values
                                                                   are matplotlib keyword arguments controlling the
                                                                   appearance of the highlighting range.
        x_points (Optional[Sequence[float]]): A sequence of x-values where the function(s) will be evaluated.
                                              If None, it defaults to 100 evenly spaced points between -1 and 1.
        title (Optional[str]): A title for the plot. If None, no title is added.
        plot (bool): Whether to display the plot. If False, the plot will not be shown but the axes will be returned.

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

    if axes is None:
        fig = plt.figure()
        if not plt_imag:
            axes = [fig.add_subplot(111)]
        else:
            axes = fig.subplots(2)
    elif not hasattr(axes, '__getitem__'):
        axes = [axes]
    if plt_imag and len(axes) != 2:
        raise ValueError("Axes must have 2 axes for imaginary plots.")

    if not plt_imag:
        plot_info = dict()
        for func_name, func in func_list.items():
            if plot_options is not None and func_name in plot_options:
                kwargs = plot_options[func_name]
            else:
                kwargs = dict()
            plot_info[func_name] = axes[0].plot(x_points, y_points[func_name].real, label=func_name, **kwargs)
        if range_plot is not None:
            for func_name, (y_min, y_max) in range_plot.items():
                if range_plot_options is not None and func_name in range_plot_options:
                    kwargs = range_plot_options[func_name]
                else:
                    kwargs = dict()
                if func_name in func_list:
                    color = plot_info[func_name][0].get_color()
                    axes[0].fill_between(x_points, y_points[func_name].real + y_min, y_points[func_name].real + y_max,
                                         color=color, alpha=0.2, **kwargs)
                else:
                    axes[0].fill_between(x_points, y_points[func_name].real + y_min, y_points[func_name].real + y_max,
                                         label=func_name, alpha=0.2, **kwargs)
        axes[0].legend()
    else:
        plot_info = dict()
        for func_name, func in func_list.items():
            if plot_options is not None and func_name in plot_options:
                kwargs = plot_options[func_name]
            else:
                kwargs = dict()
            plot_info[func_name] = (
                axes[0].plot(x_points, y_points[func_name].real, label=func_name, **kwargs),
                axes[1].plot(x_points, y_points[func_name].imag, label=func_name, **kwargs)
            )
        if range_plot is not None:
            for func_name, (y_min, y_max) in range_plot.items():
                if range_plot_options is not None and func_name in range_plot_options:
                    kwargs = range_plot_options[func_name]
                else:
                    kwargs = dict()
                if func_name in func_list:
                    color_re = plot_info[func_name][0][0].get_color()
                    color_im = plot_info[func_name][1][0].get_color()
                    axes[0].fill_between(x_points, y_points[func_name].real + y_min, y_points[func_name].real + y_max,
                                         color=color_re, alpha=0.2, **kwargs)
                    axes[1].fill_between(x_points, y_points[func_name].imag + y_min, y_points[func_name].imag + y_max,
                                         color=color_im, alpha=0.2, **kwargs)
                else:
                    axes[0].fill_between(x_points, y_points[func_name].real + y_min, y_points[func_name].real + y_max,
                                         label=func_name, alpha=0.2, **kwargs)
                    axes[1].fill_between(x_points, y_points[func_name].imag + y_min, y_points[func_name].imag + y_max,
                                         label=func_name, alpha=0.2, **kwargs)
        axes[1].legend()
    if title is not None:
        plt.title(title)
    if plot:
        plt.show()
    return axes
