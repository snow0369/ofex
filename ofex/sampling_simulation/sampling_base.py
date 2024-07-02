import math
import pickle
from itertools import product
from typing import List, Any, Optional, Union, Dict

import numpy as np
from openfermion.config import EQ_TOLERANCE

from ofex.exceptions import OfexTypeError


class ProbDist(dict):
    # _cumulative: np.ndarray
    # _cumulative_events: List[Any]

    _true_average: Optional[Any] = None
    _true_variance: Optional[Any] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isclose(sum(self.values()), 1.0, atol=EQ_TOLERANCE):
            raise ValueError("Not completed prob.", sum(self.values()))
        self.event = list(self.keys())
        self.prob = [self[k] for k in self.event]

    def sample_num(self, shots: int = 1, seed=None) -> Dict[Any, float]:
        if seed is not None:
            np.random.seed(seed)
        try:
            dist = np.random.multinomial(shots, self.prob, size=1)[0]
        except ValueError as e:
            print(shots)
            raise e
        return {k: n for k, n in zip(self.event, dist)}

    @property
    def true_average(self):
        if self._true_average is None:
            ret = 0.0
            for ev, prob in self.items():
                ret += ev * prob
            self._true_average = ret
        return self._true_average

    @property
    def true_variance(self):
        if self._true_variance is None:
            avg = 0.0
            sq_avg = 0.0
            for ev, prob in self.items():
                avg += ev * prob
                sq_avg += (ev ** 2) * prob
            self._true_variance = sq_avg - (avg ** 2)
        return self._true_variance

    @property
    def true_std(self):
        return np.sqrt(self.true_variance)

    def empirical_average(self, shots: Union[int, float], seed=None):
        if math.isinf(shots):
            return self.true_average
        elif isinstance(shots, float):
            shots = int(shots)
        if shots == 0:
            raise NotImplementedError
        samples = self.sample_num(shots, seed)
        avg = 0.0
        for ev, occ in samples.items():
            avg += ev * occ
        return avg / shots

    def empirical_variance(self, shots: int, seed=None):
        if math.isinf(shots):
            return self.true_variance
        elif isinstance(shots, float):
            shots = int(shots)
        if shots == 0:
            raise NotImplementedError
        samples = self.sample_num(shots, seed)
        avg = 0.0
        sq_avg = 0.0
        for ev, occ in samples.items():
            avg += ev * occ
            sq_avg += (ev ** 2) * occ
        avg /= shots
        sq_avg /= shots
        return sq_avg - (avg ** 2)

    def empirical_std(self, shots: int, seed=None):
        return np.sqrt(self.empirical_variance(shots, seed))

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return cls(obj)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(self), f)


class JointProbDist(ProbDist):
    _true_covariance = None

    def __init__(self, keywords: List, distr, dtype=float):
        super().__init__(distr)
        for ev in self.event:
            if not isinstance(ev, tuple):
                raise OfexTypeError(ev)
            if len(ev) != len(keywords):
                raise ValueError(f"The length of the keywords to describe the events are not the same.")
        self.keywords = keywords
        self.dtype = dtype

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            keywords, distr, dtype = pickle.load(f)
        if 'float' in dtype:
            dtype = float
        elif 'complex' in dtype:
            dtype = complex
        elif 'int' in dtype:
            dtype = int
        else:
            raise ValueError(f"Invalid dtype {dtype}")
        return cls(keywords, distr, dtype)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.keywords, dict(self), str(self.dtype)), f)

    @property
    def true_average(self):
        if self._true_average is None:
            ret = np.zeros(len(self.keywords), dtype=self.dtype)
            for ev, prob in self.items():
                ret += np.array(ev) * prob
            self._true_average = {k: val for k, val in zip(self.keywords, ret)}
        return self._true_average

    @property
    def true_variance(self):
        if self._true_variance is None:
            self._true_variance = {k: self.true_covariance[(k, k)] for k in self.keywords}
        return self._true_variance

    @property
    def true_covariance(self):
        if self._true_covariance is None:
            cov_mat = np.zeros((len(self.keywords), len(self.keywords)), dtype=self.dtype)
            for ev, prob in self.items():
                cov_mat += np.array([[ev1 * ev2 * prob for ev1 in ev]
                                     for ev2 in ev])
            cov_mat -= np.array([[self.true_average[k1] * self.true_average[k2]
                                  for k1 in self.keywords] for k2 in self.keywords])
            self._true_covariance = {(k1, k2): cov_mat[i, j]
                                     for (i, k1), (j, k2) in product(enumerate(self.keywords),
                                                                     repeat=2)}
        return self._true_covariance

    @property
    def true_std(self):
        return {k: np.sqrt(x) for k, x in self.true_variance}

    def empirical_average(self, shots: Union[int, float], seed=None):
        if math.isinf(shots):
            return self.true_average
        elif isinstance(shots, float):
            shots = int(shots)
        if shots == 0:
            return {k: 0.0 for k in self.keywords}
        samples = self.sample_num(shots, seed)
        avg = np.zeros(len(self.keywords), dtype=self.dtype)
        for ev, occ in samples.items():
            avg += np.array(ev) * occ
        return {k: val / shots for k, val in zip(self.keywords, avg)}

    def empirical_covariance(self, trial: int, seed=None):
        if math.isinf(trial):
            return self.true_covariance
        elif isinstance(trial, float):
            trial = int(trial)
        if trial == 0:
            raise NotImplementedError
        samples = self.sample_num(trial, seed)
        marg_avg = np.zeros(len(self.keywords), dtype=self.dtype)
        for ev, occ in samples.items():
            marg_avg += np.array(ev) * occ / trial
        cov_mat = np.zeros((len(self.keywords), len(self.keywords)), dtype=self.dtype)
        for ev, occ in samples.items():
            cov_mat += np.array([[ev1 * ev2 * occ / trial for ev1 in ev]
                                 for ev2 in ev])
        cov_mat -= np.array([[marg_avg[k1] * marg_avg[k2]
                              for k1 in range(len(self.keywords))]
                             for k2 in range(len(self.keywords))])
        return {(k1, k2): cov_mat[i, j] for (i, k1), (j, k2) in
                product(enumerate(self.keywords), repeat=2)}

    def empirical_variance(self, shots: int, seed=None):
        if math.isinf(shots):
            return self.true_variance
        elif isinstance(shots, float):
            shots = int(shots)
        if shots == 0:
            raise NotImplementedError
        samples = self.sample_num(shots, seed)
        avgs = np.zeros(len(self.keywords), dtype=self.dtype)
        for ev, occ in samples.items():
            avgs += np.array(ev) * occ
        avgs = avgs / shots

        variance = np.zeros(len(self.keywords), dtype=self.dtype)
        for ev, occ in samples.items():
            variance += np.array(ev) ** 2 * occ
        variance = variance / shots
        variance -= avgs ** 2
        return {k: variance[i] for i, k in enumerate(self.keywords)}

    def empirical_std(self, shots: int, seed=None):
        return {k: np.sqrt(v) for k, v in self.empirical_covariance(shots, seed).items()}


def pkl_save_prob_lst(prob: List[ProbDist], filename: str):
    with open(filename, "wb") as f:
        pickle.dump(prob, f)


def pkl_load_prob_lst(filename) -> List[ProbDist]:
    with open(filename, "rb") as f:
        dict_lst = pickle.load(f)
    return [ProbDist(x) for x in dict_lst]


if __name__ == "__main__":
    def probdist_test():
        a = ProbDist({1: 0.1,
                      2: 0.2,
                      3: 0.3,
                      4: 0.4})
        print(a.sample_num(shots=6, seed=10))
        print("Empirical")
        print(f"avg : {a.empirical_average(shots=1000_000)}")  # 3.0
        print(f"var : {a.empirical_variance(shots=1000_000)}")  # 1.0
        print("True")
        print(f"avg : {a.true_average}")  # 3.0
        print(f"var : {a.true_variance}")  # 1.0

        pkl_save_prob_lst([a], "./tmp.pkl")
        a_l = pkl_load_prob_lst("./tmp.pkl")
        print(len(a_l))
        a_l = a_l[0]
        print(a_l)


    def jointprobdist_test():
        name_var = ["v1", "v2", "v3"]
        pd = JointProbDist(
            keywords=name_var,
            distr={(1, -1, 1): 0.1,
                   (1, -2, -1): 0.2,
                   (-1, -1, -1): 0.3,
                   (1, 1, 1): 0.4})
        print("Empirical")
        print(pd.sample_num(shots=10, seed=10))
        print("AVG")
        print(pd.empirical_average(shots=1000_000))
        print("COV")
        print(pd.empirical_covariance(trial=1000_000))
        print("VAR")
        print(pd.empirical_variance(shots=1000_000))
        print("============================")
        print("TRUE")
        print("AVG")
        print(pd.true_average)
        print("COV")
        print(pd.true_covariance)
        print("VAR")
        print(pd.true_variance)


    jointprobdist_test()
