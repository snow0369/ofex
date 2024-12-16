"""
This module provides a collection of utility functions for working with nested dictionaries.
It includes functions for recursive operations such as updating, traversing, and assigning
values within nested dictionaries. The module also supports mathematical operations like
addition and subtraction of dictionary values, as well as comparison utilities that handle
both numeric and non-numeric data types.
"""

from copy import deepcopy
from numbers import Number
from typing import Optional, Any, Callable

import numpy as np
from openfermion.config import EQ_TOLERANCE

__all__ = ["recursive_dict_update", "recursive_dict_keys", "recursive_dict_items",
           "nested_dict_assign", "add_values", "sub_values", "dict_allclose", "compare_dict"]


def recursive_dict_update(previous: dict, additional: dict) -> dict:
    """
    Recursively updates a dictionary with values from another dictionary.

    Args:
        previous (dict): The original dictionary to be updated.
        additional (dict): The dictionary containing new values.

    Returns:
        dict: A new dictionary with the values from `additional` merged into `previous`.
    """
    previous = deepcopy(previous)
    for k in additional:
        add_v = additional[k]
        if k not in previous:
            previous.update({k: add_v})
            continue
        if isinstance(add_v, dict):
            previous[k] = recursive_dict_update(previous[k], add_v)
        else:
            previous[k] = add_v
    return previous


def nested_dict_assign(dictionary: dict, key_list: list, value: Any):
    """
    Assigns a value to a nested key in a dictionary.

    Args:
        dictionary (dict): The dictionary to modify.
        key_list (list): A list of keys defining the nested structure.
        value (Any): The value to assign to the nested key.
    """
    prev_key = key_list[0]
    prev_dict = dictionary
    for k in key_list[1:]:
        if prev_key not in prev_dict:
            prev_dict[prev_key] = dict()
        prev_dict = prev_dict[prev_key]
        prev_key = k
    prev_dict[prev_key] = value


def recursive_dict_keys(dictionary: dict, max_depth=None):
    """
    Recursively yields all keys in a nested dictionary.

    Args:
        dictionary (dict): The dictionary whose keys to traverse.
        max_depth (int, optional): The maximum depth to traverse. If None, no limit.

    Yields:
        tuple: The keys in the dictionary as nested tuples.
    """
    def _get_key(key, _):
        return key

    yield from _recursive_dict_items(dictionary, (), max_depth, _get_key)


def recursive_dict_items(dictionary: dict, max_depth=None):
    """
    Recursively yields all key-value pairs in a nested dictionary.

    Args:
        dictionary (dict): The dictionary whose items to traverse.
        max_depth (int, optional): The maximum depth to traverse. If None, no limit.

    Yields:
        tuple: A tuple containing the key path and the value.
    """
    def _get_item(key, value):
        return key, value

    yield from _recursive_dict_items(dictionary, (), max_depth, _get_item)


def _recursive_dict_items(dictionary: dict, current_keys, max_depth, f: Callable):
    """
    Helper function used to recursively traverse a nested dictionary.

    Args:
        dictionary (dict): The dictionary to traverse.
        current_keys (tuple): The current path of keys being traversed.
        max_depth (int, optional): The maximum depth for traversal.
        f (Callable): A function to format the keys and values.

    Yields:
        Any: Processed key-value pairs based on `f`.
    """
    for key, value in dictionary.items():
        new_keys = current_keys + (key,)
        if isinstance(value, dict) and (max_depth is None or len(new_keys) < max_depth):
            yield from _recursive_dict_items(value, new_keys, max_depth, f)
        else:
            yield f(new_keys, value)


def add_values(a: Optional[dict], b: Optional[dict]) -> Optional[dict]:
    """
    Adds the values of two dictionaries by matching their keys.

    Args:
        a (dict or None): The first dictionary with numeric values.
        b (dict or None): The second dictionary with numeric values.

    Returns:
        dict or None: A new dictionary containing the sum of values for matching keys, or None if both are None.
    """
    if a is None and b is None:
        return None
    elif a is None:
        return dict(b)
    elif b is None:
        return dict(a)
    ret = dict(a)
    for k, v in b.items():
        if k in ret:
            ret[k] += v
        else:
            ret[k] = v
    return ret


def sub_values(a: Optional[dict], b: Optional[dict]) -> Optional[dict]:
    """
    Subtracts the values of the second dictionary from the first dictionary by matching keys.

    Args:
        a (dict or None): The first dictionary with numeric values.
        b (dict or None): The second dictionary with numeric values.

    Returns:
        dict or None: A new dictionary containing the difference of values for matching keys, or None if both are None.
    """
    if a is None and b is None:
        return None
    elif a is None:
        for k, v in b.items():
            b[k] = -v
        return dict(b)
    elif b is None:
        return dict(a)
    ret = dict(a)
    for k, v in b.items():
        if k in ret:
            ret[k] -= v
        else:
            ret[k] = -v
    return ret


def dict_allclose(a: dict, b: dict, atol=EQ_TOLERANCE) -> bool:
    """
    Checks whether two dictionaries are approximately equal, element-wise.

    Args:
        a (dict): The first dictionary to compare.
        b (dict): The second dictionary to compare.
        atol (float): The absolute tolerance for comparison of numerical values.

    Returns:
        bool: True if all values are close within the given tolerance, False otherwise.
    """
    for k in set(a.keys()).union(b.keys()):
        if k not in a.keys() or k not in b.keys():
            return False
        va, vb = a[k], b[k]
        if isinstance(va, Number) and isinstance(vb, Number):
            if not np.isclose(va, vb, atol=atol):
                return False
        elif isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
            if not np.allclose(va, vb):
                return False
        elif va != vb:
            return False
    return True


def compare_dict(dict_1, dict_2,
                 repr_func: Callable[[Any, Number], str],
                 str_len=40,
                 atol=EQ_TOLERANCE) -> str:
    """
    Compares two dictionaries and generates a formatted string of differences.

    Args:
        dict_1 (dict): The first dictionary to compare.
        dict_2 (dict): The second dictionary to compare.
        repr_func (Callable): A function to format the key-value pairs for display.
        str_len (int): The length of the formatted string for each dictionary entry.
        atol (float): The absolute tolerance for numerical comparisons.

    Returns:
        str: A multi-line string summarizing the differences between `dict_1` and `dict_2`.
    """
    keys = set(dict_1.keys()).union(dict_2.keys())
    try:
        keys = sorted(keys)
    except TypeError:
        pass
    str_list = list()
    for k in keys:
        if k in dict_1:
            coeff1 = dict_1[k]
            op1_k_str = repr_func(k, coeff1).ljust(str_len)
        else:
            coeff1 = 0.0
            op1_k_str = ' ' * str_len
        if k in dict_2:
            coeff2 = dict_2[k]
            op2_k_str = repr_func(k, coeff2).ljust(str_len)
        else:
            coeff2 = 0.0
            op2_k_str = ' ' * str_len
        str_list.append(op1_k_str + ' | ' + op2_k_str + ' | ' + (f'(not matched, diff={coeff2 - coeff1})'
                                                                 if not np.isclose(coeff1, coeff2, atol=atol) else ''))
    return '\n'.join(str_list)
