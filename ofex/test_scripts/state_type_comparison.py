import os
import pickle
from itertools import product
from time import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import colormaps
from openfermion import get_sparse_operator, generate_linear_qubit_operator
from openfermion.config import EQ_TOLERANCE

from ofex.linalg.sparse_tools import apply_operator, sparse_apply_operator
from ofex.state.state_tools import to_dense, pretty_print_state, allclose, compare_states
from ofex.test_scripts.random_object import random_state_all, random_qubit_operator
import warnings

from ofex.utils.dict_utils import nested_dict_assign


def benchmark_operator_state_mult(n_qubits,
                                  density_state,
                                  operator_weight,
                                  n_trial):
    if n_trial < 1:
        raise ValueError

    state_dict, state_np, state_sp = random_state_all(n_qubits, density_state)
    operator = random_qubit_operator(n_qubits, operator_weight, hermitian=True)

    states = {"dict": state_dict, "np": state_np, "sp": state_sp}

    time_conv_results = dict()
    linear_operator, matrix = None, None

    time_conv = time()
    for _ in range(n_trial):
        linear_operator = generate_linear_qubit_operator(operator, n_qubits)
    time_conv_results["QL"] = (time() - time_conv) / n_trial
    print(f"\tConversion (Q -> L) avg. : {time_conv_results['QL']}")

    time_conv = time()
    for _ in range(n_trial):
        matrix = get_sparse_operator(operator, n_qubits)
    time_conv_results["QM"] = (time() - time_conv) / n_trial
    print(f"\tConversion (Q -> M) avg. : {time_conv_results['QM']}")

    operators = {"QubitOperator_True": operator, "QubitOperator_False": operator,
                 "LinearOperator": linear_operator, "Matrix": matrix}

    time_mul_results = dict()
    mul_results = dict()
    for (key_st, st), (key_op, op) in product(states.items(), operators.items()):
        key_tot = (key_st, key_op)
        sp = "True" in key_op
        mul_out = None
        time_mul = time()
        for _ in range(n_trial):
            if sp:
                mul_out = sparse_apply_operator(op, st)
            else:
                mul_out = apply_operator(op, st)
        time_mul_results[key_tot] = (time() - time_mul) / n_trial
        print(f"\t{key_op} -> {key_st}, sparse={sp} avg. : {time_mul_results[key_tot]}")
        mul_results[key_tot] = mul_out

    standard_name1 = ('np', 'QubitOperator_False')
    standard_state1 = mul_results[standard_name1]
    standard_name2 = ('np', 'LinearOperator')
    standard_state2 = mul_results[standard_name2]
    for idx_key, (key_tot, mul_out) in enumerate(mul_results.items()):
        try:
            assert allclose(standard_state1, mul_out, atol=EQ_TOLERANCE)
            assert allclose(standard_state2, mul_out, atol=EQ_TOLERANCE)
        except AssertionError as e:
            print(operator)
            print(state_dict)
            print(state_np)
            print('=========')
            print(to_dense(mul_out))
            print(to_dense(standard_state1))
            print(to_dense(standard_state2))
            print('=========')
            print(f'{standard_name1} <-> {key_tot}\n' + compare_states(standard_state1, mul_out))
            print(f'{standard_name2} <-> {key_tot}\n' + compare_states(standard_state2, mul_out))
            raise e

    return time_conv_results, time_mul_results


if __name__ == "__main__":
    run_simulation = True
    run_plot = True
    n_qubit_list = [10]  # [2, 4, 6, 8, 10]
    weight_op_list = [1, 10, 100, 1000]

    fname = "./state_type_comparison_result.pkl"

    if run_simulation:
        warnings.filterwarnings("error")
        time_data = dict()
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                time_data = pickle.load(f)

        for _n_qubits, _weight_operator in product(n_qubit_list, weight_op_list):
            if _weight_operator > 4 ** _n_qubits:
                continue
            for _density_state in [2 ** x for x in range(_n_qubits + 1)]:
                print(f"n_qubits: {_n_qubits}, weight_operator: {_weight_operator}, density_state: {_density_state}")
                t_conv, t_mul = benchmark_operator_state_mult(_n_qubits, _density_state, _weight_operator,
                                                              n_trial=1000)
                print('')
                for k, data in t_mul.items():
                    nested_dict_assign(time_data,
                                       ['time_mul', k, _n_qubits, _weight_operator, _density_state], data)
                for k, data in t_conv.items():
                    nested_dict_assign(time_data,
                                       ['time_conv', k, _n_qubits, _weight_operator, _density_state], data)

            with open(fname, "wb") as f:
                pickle.dump(time_data, f)

    if run_plot:  # x-axis = density_state
        colors = colormaps['tab10'](range(10))
        with open(fname, "rb") as f:
            time_data = pickle.load(f)

        n_rows = int(np.ceil(np.sqrt(len(n_qubit_list))))
        n_cols = n_rows
        fig, axes = plt.subplots(n_cols, n_rows, figsize=(n_cols * 6, n_rows * 6))
        for idx_plt, _n_qubits in enumerate(n_qubit_list):
            if len(n_qubit_list) == 1:
                ax = axes
            else:
                ax = axes.flat[idx_plt]
            for t_type, color in zip(time_data['time_mul'].keys(), colors):
                state_type, op_type = t_type
                for _weight_operator, lstyle in zip(time_data['time_mul'][_n_qubits].keys(),
                                                    ['solid', 'dashed', 'dashdot', 'dotted']):
                    tmul_data_now = time_data['time_mul'][t_type][_n_qubits][_weight_operator]
                    x_data = np.array(tmul_data_now)
                    y_data = np.array([tmul_data_now[x] for x in x_data])
                    tconv_ql_data_now = np.array([time_data['time_conv']["QL"][_n_qubits][_weight_operator][x]
                                                  for x in x_data])
                    tconv_qm_data_now = np.array([time_data['time_conv']["QM"][_n_qubits][_weight_operator][x]
                                                  for x in x_data])
                    if op_type == 'LinearOperator':
                        y_data += tconv_ql_data_now
                    elif op_type == 'Matrix':
                        y_data += tconv_qm_data_now
                    elif op_type in ["QubitOperator_True", "QubitOperator_False"]:
                        pass
                    else:
                        raise ValueError
                    ax.plot(x_data, y_data,
                            label=f"{t_type}, weight_operator = {_weight_operator}", linestyle=lstyle, color=color)
            ax.set_xlabel(f"Density_State")
            ax.set_ylabel("Time")
            ax.set_title(f"n_qubits = {_n_qubits}")
            ax.set_yscale('log')
            if idx_plt == 0:
                ax.legend()
            plt.show()
