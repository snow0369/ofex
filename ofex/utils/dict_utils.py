from copy import deepcopy
from numbers import Number
from typing import Optional, Any, Callable

import numpy as np
from openfermion.config import EQ_TOLERANCE


def recursive_dict_update(previous: dict, additional: dict) -> dict:
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
    prev_key = key_list[0]
    prev_dict = dictionary
    for k in key_list[1:]:
        if prev_key not in prev_dict:
            prev_dict[prev_key] = dict()
        prev_dict = prev_dict[prev_key]
        prev_key = k
    prev_dict[prev_key] = value


def recursive_dict_keys(dictionary: dict, max_depth=None):
    def _get_key(key, value):
        return key

    yield from _recursive_dict_items(dictionary, (), max_depth, _get_key)


def recursive_dict_items(dictionary: dict, max_depth=None):
    def _get_item(key, value):
        return key, value

    yield from _recursive_dict_items(dictionary, (), max_depth, _get_item)


def _recursive_dict_items(dictionary: dict, current_keys, max_depth, f: Callable):
    for key, value in dictionary.items():
        new_keys = current_keys + (key,)
        if isinstance(value, dict) and (max_depth is None or len(new_keys) < max_depth):
            yield from _recursive_dict_items(value, new_keys, max_depth, f)
        else:
            yield f(new_keys, value)


def add_values(a: Optional[dict], b: Optional[dict]) -> Optional[dict]:
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
