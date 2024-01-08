import nltk
import itertools
import numpy as np
from abc import ABCMeta, abstractmethod
from multiprocessing import Manager, Process


class BaseModel(metaclass=ABCMeta):

    def __init__(
            self,
            order=2,
            n_jobs=1
    ):
        self.order = order
        self.n_jobs = n_jobs
        self.states = None
        self.observations = None
        self.A = None
        self.B = None
        self.states_population = None
        self.observations_population = None
        self.decode_proba = None

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def _transition_proba(self, ngram):
        pass

    @abstractmethod
    def _emission_proba(self, observation, state):
        pass

    def _fit(self, X, y):
        self.states = self.value_counts(y)
        self.states_population = self.everygrams(y, order=self.order)
        self._fit_transition(self.states_population, self.order)

        self.observations = self.value_counts(X)
        self.observations_population = self.populate_observations(X, y)
        self._fit_emission(self.observations_population)

        return self

    def _fit_transition(self, states_population, order):
        self.A = {
            ngram: self._transition_proba(ngram) for ngram in states_population if len(ngram) == order
        }

    def _fit_emission(self, observations_population):
        self.B = {
            observation: {
                state: self._emission_proba(observation, state) for state in observation_states
            } for observation, observation_states in observations_population.items()
        }

    def _predict(self, X, return_proba=False):
        self.decode_proba, y_pred = self._multi_process_predict(X, n_processes=self.n_jobs)
        return self.decode_proba if return_proba else y_pred

    def _single_process_predict(self, X, process_notepad, process_id):
        process_notepad[process_id] = [self._viterbi(sequence) if sequence else (0, []) for sequence in X]

    def _multi_process_predict(self, X, n_processes):
        X = self._chunks(X, n_processes)
        predicted = []
        with Manager() as manager:
            notepad = manager.dict()
            processes = [
                Process(target=self._single_process_predict, args=(next(X), notepad, i)) for i in range(n_processes)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            for i in range(n_processes):
                predicted.extend(notepad[i])
        return zip(*predicted)

    def _viterbi(self, sequence):
        pi, bp = self._viterbi_forward(sequence)
        return self._viterbi_backward(sequence, pi, bp)

    def _viterbi_forward(self, sequence):
        pi, bp = {(-1, *('<s>',) * (self.order - 1)): 0}, {}
        pad_sequence = [*('<s>',) * (self.order - 1), *sequence, '</s>']
        for i, obs in enumerate(pad_sequence[self.order-1:-1], start=0):
            states_product = self.multiple_get_states(pad_sequence[i:i+self.order])
            step = len(self.get_states(pad_sequence[i]))
            for j in range(0, len(states_product), step):
                scores = []
                for tup in states_product[j:j+step]:
                    scores.append(
                        pi[(i-1, *tup[:self.order-1])] +
                        self.get_transition_proba(tup) + self.get_emission_proba(obs, tup[-1])
                    )
                best_tuple = states_product[j:j+step][np.argmax(scores)]
                pi[(i, *best_tuple[-(self.order-1):])] = np.max(scores)
                bp[(i, *best_tuple[-(self.order-1):])] = best_tuple[:self.order-1]
        return pi, bp

    def _viterbi_backward(self, sequence, pi, bp):
        sequence = ["<s>", *sequence]
        back_pointers = []
        for step in reversed(range(len(sequence))):
            if step == len(sequence) - 1:
                states_product = self.multiple_get_states([*sequence[step-self.order+2:], '</s>'])
                scores = [
                    pi[(step-1, *tup[:-1])] + self.get_transition_proba(tup) for tup in states_product
                ]
                best_score, best_tuple = np.max(scores), states_product[np.argmax(scores)]
                back_pointers.append(best_tuple[:-1])
            else:
                back_pointers.insert(0, bp[(step, *back_pointers[0])])
        y_pred = [b[-1] for b in back_pointers[1:]]
        return best_score, y_pred

    def get_states(self, observation):
        if observation == "<s>" or observation == "</s>":
            return [observation]
        return list(self.observations_population.get(observation, self.states))

    def multiple_get_states(self, observations):
        states = [self.get_states(observation) for observation in reversed(observations)]
        return [state_product[::-1] for state_product in itertools.product(*states)]

    def get_transition_proba(self, ngram):
        return np.log(self.A.get(ngram, 1e-16))

    def get_emission_proba(self, observation, state):
        return np.log(self.B.get(observation, {}).get(state, 1e-16))

    @property
    def decode_proba_(self):
        return np.exp(self.decode_proba)

    @property
    def decode_log_proba_(self):
        return np.array(self.decode_proba)

    @staticmethod
    def value_counts(sequences):
        counts = {}
        for sequence in sequences:
            for element in sequence:
                counts[element] = counts.get(element, 0) + 1
        return counts

    @staticmethod
    def everygrams(sequences, order, left_pad='<s>', right_pad='</s>'):
        ngrams = {}
        for sequence in sequences:
            for n in range(order, 0, -1):
                for ngram in nltk.ngrams([*(left_pad,)*n, *sequence, right_pad], n):
                    ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    @staticmethod
    def populate_observations(obs_sequences, states_sequences):
        population = {}
        for obs_sequence, state_sequence in zip(obs_sequences, states_sequences):
            for observation, state in zip(obs_sequence, state_sequence):
                if observation not in population:
                    population[observation] = {}
                population[observation][state] = population[observation].get(state, 0) + 1
        return population

    @staticmethod
    def _chunks(X, n):
        k, m = divmod(len(X), n)
        return (X[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class BaseHMM(BaseModel):

    @abstractmethod
    def __init__(
            self,
            order=2,
            n_jobs=1
    ):
        super().__init__(
            order=order,
            n_jobs=n_jobs
        )

    def _transition_proba(self, ngram):
        return self.states_population[ngram] / self.states_population[ngram[:-1]]

    def _emission_proba(self, observation, state):
        return self.observations_population[observation][state] / self.states[state]


class HMM(BaseHMM):

    def __init__(
            self,
            order=2,
            n_jobs=1
    ):
        super().__init__(
            order=order,
            n_jobs=n_jobs
        )

    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X, return_proba=False):
        return self._predict(X, return_proba)


class BaseSmoothHMM(BaseModel):

    @abstractmethod
    def __init__(
            self,
            order=2,
            n_jobs=1
    ):
        super().__init__(
            order=order,
            n_jobs=n_jobs
        )
        self.prior_A = None
        self.prior_B = None
        self.singleton_A = None
        self.singleton_B = None

    def _fit_transition(self, states_population, order):
        self.prior_A = self._transition_prior(self.states)
        self.singleton_A = self._transition_singleton(self.states_population)
        self.A = {
            ngram: self._transition_proba(ngram) for ngram in self.states_population if len(ngram) == self.order
        }

    def _transition_proba(self, ngram):
        if len(ngram) == 1:
            return self.prior_A.get(ngram[0], 1e-6)
        C = 1 + self.singleton_A.get(ngram[:-1], 0)
        return (
            (self.states_population.get(ngram, 0) + C * self._transition_proba(ngram[-(len(ngram) - 1):])) /
            (self.states_population.get(ngram[:-1], 0) + C)
        )

    @staticmethod
    def _transition_prior(states):
        prior_transition = {}
        for state in states:
            prior_transition[state] = (1 + states[state]) / (len(states) + sum(states.values()))
        return prior_transition

    @staticmethod
    def _transition_singleton(states_population):
        singletons = {}
        for ngram, count in states_population.items():
            if count == 1 and len(ngram) > 1:
                singletons[ngram[:-1]] = singletons.get(ngram[:-1], 0) + 1
        return singletons

    def _fit_emission(self, observations_population):
        self.prior_B = self._emission_prior(self.states, self.observations_population)
        self.singleton_B = self._emission_singleton(self.observations_population)
        self.B = {
            observation: {
                state: self._emission_proba(observation, state) for state in observation_states
            } for observation, observation_states in observations_population.items()
        }

    def _emission_proba(self, observation, state):
        C = 1 + self.singleton_B.get(state, 0)
        return (
            (self.observations_population.get(observation, {}).get(state, 0) +
             C * self.prior_B.get(observation, 1 / (len(self.states) + sum(self.states.values())))) /
            (self.states_population[(state,)] + C)
        )

    @staticmethod
    def _emission_prior(states, obs_population):
        prior_emission = {}
        for observation, observation_states in obs_population.items():
            prior_emission[observation] = sum(observation_states.values()) / sum(states.values())
        return prior_emission

    @staticmethod
    def _emission_singleton(obs_population):
        singletons = {}
        for observation, observation_states in obs_population.items():
            for state, count in observation_states.items():
                if count == 1:
                    singletons[state] = singletons.get(state, 0) + 1
        return singletons

    def get_transition_proba(self, ngram):
        return np.log(self.A.get(ngram, self._transition_proba(ngram)))

    def get_emission_proba(self, observation, state):
        return np.log(self.B.get(observation, {}).get(state, self._emission_proba(observation, state)))


class SmoothHMM(BaseSmoothHMM):

    def __init__(
            self,
            order=2,
            n_jobs=1
    ):
        super().__init__(
            order=order,
            n_jobs=n_jobs
        )

    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X, return_proba=False):
        return self._predict(X, return_proba)
