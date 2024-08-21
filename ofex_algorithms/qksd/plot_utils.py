import os
import pickle
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from ofex_algorithms.qksd.qksd_utils import trunc_eigh

from matplotlib import pyplot as plt

fci_energy_list = {
    "H2O": 0.00
}


def parse_const(fname, shift_lvl):
    with open(fname, 'rb') as f:
        fham_t, f_const, pham_t, p_const, cisd_state, shift_op_pikl = pickle.load(f)
    return f_const + p_const, shift_op_pikl[shift_lvl][-1]


def cut_list(distr: np.ndarray, population=0.9):
    distr = np.sort(distr)
    n = len(distr)
    st_idx = int(n * (1.0 - population) / 2)
    end_idx = n - st_idx
    return distr[st_idx:end_idx]


def parse_sample(ideal_fname, sample_fname, const, shift_const,
                 shift_lvl, mol_name):
    fci_energy = fci_energy_list[mol_name]

    with open(ideal_fname, 'rb') as f:
        ideal_matrices = pickle.load(f)
    ideal_h, ideal_s = ideal_matrices["non-shifted"]
    ideal_h, ideal_s = np.array(ideal_h), np.array(ideal_s)

    if shift_lvl is None:
        eigval, _ = trunc_eigh(ideal_h, ideal_s, epsilon=1e-12)
        ideal_energy = np.min(eigval) + const
    else:
        shift_h, shift_s = ideal_matrices[f"shifted_{shift_lvl}"]
        eigval, _ = trunc_eigh(shift_h, shift_s, epsilon=1e-12)
        ideal_energy = np.min(eigval) + const + shift_const

    with open(sample_fname, 'rb') as f:
        sample_h_list, sample_s_list = pickle.load(f)
    sample_h_list, sample_s_list = np.array(sample_h_list), np.array(sample_s_list)

    max_n = sample_h_list.shape[-1]
    sample_energies = list()
    for sample_idx in range(sample_h_list.shape[0]):
        sample_h, sample_s = sample_h_list[sample_idx, :, :], sample_s_list[sample_idx, :, :]
        tmp_sample_energy = list()
        for trunc_n in range(2, max_n + 1):
            eigval, _ = trunc_eigh(sample_h, sample_s, n_lambda=trunc_n)
            if shift_lvl is None:
                tmp_sample_energy.append(np.min(eigval) + const)
            else:
                tmp_sample_energy.append(np.min(eigval) + const + shift_const)
        tmp_sample_energy = np.array(tmp_sample_energy)
        sample_energies.append(tmp_sample_energy[np.argmin(np.abs(tmp_sample_energy - ideal_energy))])

    return sample_energies, ideal_energy, fci_energy


def color_table(len_color):
    return plt.cm.jet(np.linspace(0, 1, len_color))


def plot_histogram(ax,
                   distr_list: List[np.ndarray],
                   fci_energy,
                   title,
                   label_list,
                   population=0.9):
    distr_list = list(distr_list)
    orig_n = list()
    print("Outside Scope = ")
    for idx in range(len(distr_list)):
        distr = distr_list[idx]
        orig_n.append(len(distr))
        distr_list[idx] = cut_list(distr, population)
        print(f"\t{(orig_n[idx] - len(distr_list[idx])) / orig_n[idx] * 100} %")

    n_bin = 100
    min_x, max_x = min([np.min(distr) for distr in distr_list]), \
        max([np.max(distr) for distr in distr_list])
    bins = np.linspace(min_x, max_x, n_bin)
    colors = color_table(len(distr_list))
    for c, distr, label in zip(colors, distr_list, label_list):
        ax.hist(distr, alpha=0.5, bins=bins, density=False, color=c, label=label)
    ax.axvline(x=fci_energy, color='b', label="FCI")
    ax.legend()
    ax.set_title(title)


def plot_vanilla_sample_hist(ham_fname):
    ham_const, shift_const = parse_const(ham_fname)
    sampled_energy, ideal_value, fci_value = parse_sample(ideal_log, sample_pkl, const)
    if not os.path.isdir(output_graph_dir):
        os.mkdir(output_graph_dir)
    for trunc_n, samples in sampled_energy.items():
        fig = plt.figure()
        ax = fig.add_subplot()
        plot_histogram(ax, distr_list=[samples],
                       fci_energy=fci_value,
                       title=f"{graph_title} trunc_n = {trunc_n}",
                       label_list=["vanilla"],
                       population=0.9)
        fig.savefig(os.path.join(output_graph_dir, f"trunc_{trunc_n}.png"))
