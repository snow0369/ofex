"""
This module provides tools for representing and manipulating probability distributions.

Classes:
- ProbDist: Represents a single probability distribution. Contains methods for sampling,
  computing statistical metrics such as mean, variance, and standard deviation.
- JointProbDist: Extends ProbDist to handle multidimensional events (joint distributions),
  supporting covariance computation and other joint distribution functionalities.

These classes are designed for flexibility and support various operations related to
probabilistic modeling and numerical simulations.
"""

import math
from itertools import product
from typing import List, Any, Optional, Union, Dict, Tuple

import numpy as np
from openfermion.config import EQ_TOLERANCE

from ofex.exceptions import OfexTypeError

__all__ = ["ProbDist", "JointProbDist"]

EventType = Union[int, float, complex]

class ProbDist(dict):
    """
    Represents a probability distribution as a dictionary where keys are events and 
    values are their probabilities. Provides functionalities for sampling, computing 
    statistical moments, and related operations.
    """
    # _cumulative: np.ndarray
    # _cumulative_events: List[Any]

    _true_average: Optional[Any] = None
    _true_variance: Optional[Any] = None

    def __init__(self, *args, **kwargs):
        """
        Initialize the probability distribution by checking normalization and 
        extracting events and their probabilities.
    
        Args:
            *args: Arguments to initialize the dictionary.
            **kwargs: Keyword arguments to initialize the dictionary.
    
        Raises:
            ValueError: If the sum of probabilities is not 1 within tolerance.
        """
        super().__init__(*args, **kwargs)
        if not np.isclose(sum(self.values()), 1.0, atol=EQ_TOLERANCE):
            raise ValueError("Not completed prob.", sum(self.values()))
        self.event = list(self.keys())
        self.prob = [self[k] for k in self.event]

    def sample_num(self,
                   shots: int = 1,
                   seed: Optional[int]=None)\
            -> Dict[EventType, float]:
        """
        Sample event occurrences based on the probability distribution.
    
        Args:
            shots (int): Number of samples to draw.
            seed (Optional[int]): Seed for the random number generator.
    
        Returns:
            Dict[EventType, float]: A dictionary mapping events to occurrences.
        """
        if seed is not None:
            np.random.seed(seed)
        try:
            dist = np.random.multinomial(shots, self.prob, size=1)[0]
        except ValueError as e:
            print(shots)
            raise e
        return dict(zip(self.event, dist))

    @property
    def true_average(self) -> EventType:
        """
        Calculate the true average (mean) of the probability distribution.
    
        Returns:
            EventType: The computed true average value.
        """
        if self._true_average is None:
            ret = 0.0
            for ev, prob in self.items():
                ret += ev * prob
            self._true_average = ret
        return self._true_average

    @property
    def true_variance(self) -> float:
        """
        Calculate the true variance of the probability distribution.
    
        Returns:
            float: The computed true variance value.
        """
        if self._true_variance is None:
            avg = 0.0
            sq_avg = 0.0
            for ev, prob in self.items():
                avg += ev * prob
                sq_avg += (abs(ev) ** 2) * prob
            self._true_variance = sq_avg - (abs(avg) ** 2)
        return self._true_variance

    @property
    def true_std(self) -> float:
        """
        Calculate the true standard deviation of the probability distribution.
    
        Returns:
            float: The computed true standard deviation value.
        """
        return np.sqrt(self.true_variance)

    def empirical_average(self,
                          shots: Union[int, float],
                          seed: Optional[int]=None)\
            -> EventType:
        """
        Calculate the empirical average of the probability distribution.
    
        Args:
            shots (Union[int, float]): The number of samples to draw for computing the average. 
                                       If infinite, returns the true average.
            seed (Optional[int]): Seed for the random number generator.
    
        Returns:
            EventType: The computed empirical average value.
        """
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

    def batched_empirical_average(self,
                                  shots: Union[int, float],
                                  n_batch: int,
                                  seed: Optional[int]=None)\
            -> List[EventType]:
        """
        Compute a batched empirical average over multiple iterations.
    
        Args:
            shots (Union[int, float]): The number of samples to draw for each batch.
            n_batch (int): The number of batches to compute.
            seed (Optional[int]): Seed for the random number generator.
    
        Returns:
            List[EventType]: A list of empirical averages for each batch.
        """
        if seed is not None:
            np.random.seed(seed)
        output = list()
        for _ in range(n_batch):
            output.append(self.empirical_average(shots, seed=None))
        return output

    def empirical_variance(self,
                           shots: int,
                           seed: Optional[int]=None)\
            -> float:
        """
        Calculate the empirical variance of the probability distribution.
    
        Args:
            shots (int): The number of samples to draw for computing the variance. 
                         If infinite, returns the true variance.
            seed (Optional[int]): Seed for the random number generator.
    
        Returns:
            float: The computed empirical variance value.
        """
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
            sq_avg += (abs(ev) ** 2) * occ
        avg /= shots
        sq_avg /= shots
        return sq_avg - (abs(avg) ** 2)

    def empirical_std(self,
                      shots: int,
                      seed:Optional[int]=None)\
            -> EventType:
        """
        Compute the empirical standard deviation of the probability distribution.
    
        Args:
            shots (int): The number of samples to draw for computing the standard deviation.
            seed (Optional[int]): Seed for the random number generator.
    
        Returns:
            EventType: The computed empirical standard deviation value.
        """
        return np.sqrt(self.empirical_variance(shots, seed))

    @classmethod
    def unpickle(cls, pickled_obj: Dict[EventType, float]):
        """
        Unpickle a serialized probability distribution.
    
        Args:
            pickled_obj (Dict[EventType, float]): The serialized probability distribution object.
    
        Returns:
            ProbDist: A reconstructed ProbDist object.
        """
        return cls(pickled_obj)

    def pickle(self) -> Dict[EventType, float]:
        """
        Serialize the probability distribution into a dictionary.
    
        Returns:
            Dict[EventType, float]: A dictionary containing the serialized probability distribution.
        """
        return dict(self)


class JointProbDist(ProbDist):
    """
    Represents a joint probability distribution over multiple variables 
    with additional capabilities to compute covariance and handle multidimensional events.
    """
    _true_covariance = None

    def __init__(self,
                 keywords: List[str],
                 distr: Dict[Tuple[EventType, ...], float],
                 dtype=float):
        """
        Initialize a joint probability distribution with events and their probabilities.
    
        Args:
            keywords (List[str]): Descriptive labels for each variable in the joint distribution.
            distr (Dict[Tuple[EventType, ...], float]): Dictionary mapping multidimensional events 
                (tuples) to their associated probabilities.
            dtype (type): Data type for computations (default is float).
    
        Raises:
            OfexTypeError: If an event in the distribution is not a tuple.
            ValueError: If the number of event dimensions does not match the number of keywords.
        """
        super().__init__(distr)
        for ev in self.event:
            if not isinstance(ev, tuple):
                raise OfexTypeError(ev)
            if len(ev) != len(keywords):
                raise ValueError(f"The length of the keywords to describe the events are not the same.")
        self.keywords = keywords
        self.dtype = dtype

    @classmethod
    def unpickle(cls,
                 pickled_obj: Tuple[List[str], Dict[Tuple[EventType, ...], float], str]):
        """
        Deserialize a joint probability distribution.
    
        Args:
            pickled_obj (Tuple[List[str], Dict[Tuple[EventType, ...], float], str]): 
                The serialized object containing keywords, event probabilities, and data type.
    
        Returns:
            JointProbDist: A reconstructed JointProbDist instance.
    
        Raises:
            ValueError: If the data type string is invalid.
        """
        keywords, distr, dtype = pickled_obj
        if 'float' in dtype:
            dtype = float
        elif 'complex' in dtype:
            dtype = complex
        elif 'int' in dtype:
            dtype = int
        else:
            raise ValueError(f"Invalid dtype {dtype}")
        return cls(keywords, distr, dtype)

    def pickle(self) -> Tuple[List[str], Dict[Tuple[EventType, ...], float], str]:
        """
        Serialize the joint probability distribution into a tuple.
    
        Returns:
            Tuple[List[str], Dict[Tuple[EventType, ...], float], str]: A tuple containing 
                the keywords, event probabilities, and data type of the distribution.
        """
        return self.keywords, dict(self), str(self.dtype)

    def sample_num(self,
                   shots: int = 1,
                   seed: Optional[int] = None) \
            -> Dict[Tuple[EventType], float]:
        """
        Sample event occurrences based on the probability distribution.

        Args:
            shots (int): Number of samples to draw.
            seed (Optional[int]): Seed for the random number generator.

        Returns:
            Dict[Tuple[EventType], float]: A dictionary mapping events to occurrences.
        """
        if seed is not None:
            np.random.seed(seed)
        try:
            dist = np.random.multinomial(shots, self.prob, size=1)[0]
        except ValueError as e:
            print(shots)
            raise e
        return dict(zip(self.event, dist))

    @property
    def true_average(self) -> Dict[str, EventType]:
        """
        Compute the true average for each variable in the joint distribution.
    
        Returns:
            Dict[str, EventType]: A dictionary mapping each variable label to its mean value.
        """
        if self._true_average is None:
            ret = np.zeros(len(self.keywords), dtype=self.dtype)
            for ev, prob in self.items():
                ret += np.array(ev) * prob
            self._true_average = {k: val for k, val in zip(self.keywords, ret)}
        return self._true_average

    @property
    def true_variance(self) -> Dict[str, float]:
        """
        Compute the true variance for each variable in the joint distribution.
    
        Returns:
            Dict[str, float]: A dictionary mapping each variable label to its variance.
        """
        if self._true_variance is None:
            self._true_variance = {k: self.true_covariance[(k, k)] for k in self.keywords}
        return self._true_variance

    @property
    def true_covariance(self) -> Dict[Tuple[str, str], EventType]:
        """
        Compute the true covariance matrix over all variables in the joint distribution.
    
        Returns:
            Dict[Tuple[str, str], EventType]: A dictionary mapping pairs of variables 
            to their covariance values.
        """
        if self._true_covariance is None:
            cov_mat = np.zeros((len(self.keywords), len(self.keywords)), dtype=self.dtype)
            for ev, prob in self.items():
                cov_mat += np.array([[ev1.conjugate() * ev2 * prob for ev1 in ev]
                                     for ev2 in ev])
            cov_mat -= np.array([[self.true_average[k1].conjugate() * self.true_average[k2]
                                  for k1 in self.keywords] for k2 in self.keywords])
            self._true_covariance = {(k1, k2): cov_mat[i, j]
                                     for (i, k1), (j, k2) in product(enumerate(self.keywords),
                                                                     repeat=2)}
        return self._true_covariance

    @property
    def true_std(self) -> Dict[str, float]:
        """
        Compute the true standard deviation for each variable in the joint distribution.
    
        Returns:
            Dict[str, float]: A dictionary mapping each variable label to its standard deviation.
        """
        return {k: np.sqrt(x) for k, x in self.true_variance}

    def empirical_average(self,
                          shots: Union[int, float],
                          seed: Optional[int] = None) \
            -> Dict[str, EventType]:
        """
        Compute the empirical average for each variable based on sampled data.
    
        Args:
            shots (Union[int, float]): Number of samples to draw. If infinite, returns the true average.
            seed (Optional[int]): Seed for the random number generator.
    
        Returns:
            Dict[str, EventType]: A dictionary mapping variable labels to their empirical averages.
        """
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

    def empirical_covariance(self,
                             shots: int,
                             seed: Optional[int] = None) \
            -> Dict[Tuple[str, str], complex]:
        """
        Compute the empirical covariance matrix using sampled data.
    
        Args:
            shots (int): Number of samples to use for the computation.
            seed (Optional[int]): Seed for the random number generator.
    
        Returns:
            Dict[Tuple[str, str], EventType]: Empirical covariance between all variable pairs.
    
        Raises:
            NotImplementedError: If the number of shots is zero.
        """
        if math.isinf(shots):
            return self.true_covariance
        elif isinstance(shots, float):
            shots = int(shots)
        if shots == 0:
            raise NotImplementedError
        samples = self.sample_num(shots, seed)
        marg_avg = np.zeros(len(self.keywords), dtype=self.dtype)
        for ev, occ in samples.items():
            marg_avg += np.array(ev) * occ / shots
        cov_mat = np.zeros((len(self.keywords), len(self.keywords)), dtype=self.dtype)
        for ev, occ in samples.items():
            cov_mat += np.array([[ev1.conjugate() * ev2 * occ / shots for ev1 in ev]
                                 for ev2 in ev])
        cov_mat -= np.array([[marg_avg[k1].conjugate() * marg_avg[k2]
                              for k1 in range(len(self.keywords))]
                             for k2 in range(len(self.keywords))])
        return {(k1, k2): complex(cov_mat[i, j]) for (i, k1), (j, k2) in
                product(enumerate(self.keywords), repeat=2)}

    def empirical_variance(self,
                           shots: int,
                           seed: Optional[int] = None) \
            -> Dict[str, float]:
        """
        Compute the empirical variance of each variable.
    
        Args:
            shots (int): Number of samples to use for the computation.
            seed (Optional[int]): Seed for the random number generator.
    
        Returns:
            Dict[str, float]: A dictionary mapping each variable label to its variance.
    
        Raises:
            NotImplementedError: If the number of shots is zero.
        """
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
            variance += np.abs(np.array(ev)) ** 2 * occ
        variance = variance / shots
        variance -= abs(avgs) ** 2
        return {k: float(variance[i]) for i, k in enumerate(self.keywords)}

    def empirical_std(self,
                      shots: int,
                      seed: Optional[int] = None) \
            -> Dict[str, float]:
        """
        Compute the empirical standard deviation for each variable.
    
        Args:
            shots (int): Number of samples to use for the computation.
            seed (Optional[int]): Seed for the random number generator.
    
        Returns:
            Dict[str, float]: A dictionary mapping each variable label to its standard deviation.
        """
        return {k: np.sqrt(v) for k, v in self.empirical_variance(shots, seed).items()}
