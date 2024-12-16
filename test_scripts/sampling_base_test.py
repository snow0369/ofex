
import pickle

from ofex.sampling_simulation.sampling_base import ProbDist, JointProbDist


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

    with open("./tmp.pkl", "wb") as f:
        # noinspection PyTypeChecker
        pickle.dump(a.pickle(), f)

    with open("./tmp.pkl", "rb") as f:
        a_l = ProbDist.unpickle(pickle.load(f))
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
    print(pd.empirical_covariance(shots=1000_000))
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
