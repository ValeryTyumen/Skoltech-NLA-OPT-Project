import numpy as np
from tools import compute_frequencies

class ARTM:

    def __init__(self, topics_count, regularizers, regularizer_weights):

        self._topics_count = topics_count
        self._regularizers = regularizers
        self._regularizer_weights = regularizer_weights

    def train(self, word_in_doc_freqs, words_list, iterations_count=100, verbose=False, seed=None):
        """
            word_in_topics_freqs - let it be `sparse.dok_matrix` for now

            TODO: use log-probs?
        """

        words_count = word_in_doc_freqs.shape[0]
        docs_count = word_in_doc_freqs.shape[1]

        if seed is not None:
            np.random.seed(seed=seed)

        # \phi_{wt}
        word_in_topic_probs = np.random.uniform(size=(words_count, self._topics_count))
        word_in_topic_probs /= word_in_topic_probs.sum(axis=0)

        # \theta_{td}
        topic_in_doc_probs = np.random.uniform(size=(self._topics_count, docs_count))
        topic_in_doc_probs /= topic_in_doc_probs.sum(axis=0)

        # n_{wt}
        word_in_topics_freqs_buffer = np.zeros_like(word_in_topic_probs)
        # n_{dt}
        topic_in_doc_freqs_buffer = np.zeros_like(topic_in_doc_probs)

        loglikelihoods = []

        for iteration_index in range(iterations_count):

            self._do_em_iteration(word_in_doc_freqs, word_in_topic_probs, topic_in_doc_probs,
                    word_in_topics_freqs_buffer, topic_in_doc_freqs_buffer)

            loglikelihood = self._get_loglikelihood(word_in_doc_freqs, word_in_topic_probs,
                        topic_in_doc_probs)

            loglikelihoods.append(loglikelihood)

            if verbose:
                print('iter#{0}: loglike={1}'.format(iteration_index + 1, loglikelihood))

        return ARTMTrainResult(word_in_topic_probs, topic_in_doc_probs, words_list, loglikelihoods)

    def _do_em_iteration(self, word_in_doc_freqs, word_in_topic_probs, topic_in_doc_probs,
            word_in_topics_freqs_buffer, topic_in_doc_freqs_buffer):

        EPSILON = 1e-10

        word_in_topics_freqs, topic_in_doc_freqs, _, _ = compute_frequencies(word_in_doc_freqs,
                word_in_topic_probs, topic_in_doc_probs, word_in_topics_freqs_buffer,
                topic_in_doc_freqs_buffer) 

        unnormalized_word_in_topic_probs = word_in_topics_freqs
        unnormalized_topic_in_doc_probs = topic_in_doc_freqs

        for regularizer, weight in zip(self._regularizers, self._regularizer_weights):

            word_in_topic_probs_grad, topic_in_doc_probs_grad = \
                    regularizer.get_gradient(word_in_topic_probs, topic_in_doc_probs)

            word_in_topic_freqs_addition = word_in_topic_probs_grad
            word_in_topic_freqs_addition *= word_in_topic_probs
            word_in_topic_freqs_addition *= weight

            unnormalized_word_in_topic_probs += word_in_topic_freqs_addition

            topic_in_doc_freqs_addition = topic_in_doc_probs_grad
            topic_in_doc_freqs_addition *= topic_in_doc_probs
            topic_in_doc_freqs_addition *= weight

            unnormalized_topic_in_doc_probs += topic_in_doc_freqs_addition

        np.clip(unnormalized_word_in_topic_probs, 0, None, out=unnormalized_word_in_topic_probs)
        np.clip(unnormalized_topic_in_doc_probs, 0, None, out=unnormalized_topic_in_doc_probs)
 
        word_in_topic_prob_norm_consts = unnormalized_word_in_topic_probs.sum(axis=0)
        word_in_topic_prob_norm_const_is_not_small = (word_in_topic_prob_norm_consts > EPSILON)

        word_in_topic_probs[:, :] = unnormalized_word_in_topic_probs
        word_in_topic_probs[:, np.logical_not(word_in_topic_prob_norm_const_is_not_small)] = 0
        word_in_topic_probs[:, word_in_topic_prob_norm_const_is_not_small] /= \
                word_in_topic_prob_norm_consts[word_in_topic_prob_norm_const_is_not_small] 

        topic_in_doc_prob_norm_consts = unnormalized_topic_in_doc_probs.sum(axis=0)
        topic_in_doc_prob_norm_const_is_not_small = (topic_in_doc_prob_norm_consts > EPSILON)

        topic_in_doc_probs[:, :] = unnormalized_topic_in_doc_probs
        topic_in_doc_probs[:, np.logical_not(topic_in_doc_prob_norm_const_is_not_small)] = 0
        topic_in_doc_probs[:, topic_in_doc_prob_norm_const_is_not_small] /= \
                topic_in_doc_prob_norm_consts[topic_in_doc_prob_norm_const_is_not_small]

    def _get_loglikelihood(self, word_in_doc_freqs, word_in_topic_probs, topic_in_doc_probs):

        loglikelihood = 0.0

        for (word_index, doc_index), word_in_doc_freq in word_in_doc_freqs.items():
            loglikelihood += word_in_doc_freq*np.log(word_in_topic_probs[word_index].dot(
                    topic_in_doc_probs[:, doc_index]))

        for regularizer, weight in zip(self._regularizers, self._regularizer_weights):
            loglikelihood += weight*regularizer.get_value(word_in_topic_probs, topic_in_doc_probs)

        return loglikelihood

class ARTMTrainResult:

    def __init__(self, word_in_topic_probs, topic_in_doc_probs, words_list, loglikelihoods):

        self._word_in_topic_probs = word_in_topic_probs
        self._topic_in_doc_probs = topic_in_doc_probs
        self._words_list = words_list
        self._loglikelihoods = loglikelihoods

    @property
    def word_in_topic_probs(self):

        return self._word_in_topic_probs

    @property
    def topic_in_doc_probs(self):

        return self._topic_in_doc_probs

    @property
    def loglikelihoods(self):

        return self._loglikelihoods

    def get_top_words_in_topics(self, top_words_count):

        top_words_indices = np.argsort(self._word_in_topic_probs, axis=0)[-top_words_count:][::-1]

        return self._words_list[top_words_indices]

    def get_top_topics_in_docs(self, top_topics_count):

        return np.argsort(self._topic_in_doc_probs, axis=0)[-top_topics_count:][::-1]
