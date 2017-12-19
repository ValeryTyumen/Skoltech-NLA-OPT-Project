import numpy as np
from scipy import sparse
from tools import compute_frequencies

class ARTM:

    def __init__(self, topics_count, regularizers, regularizer_weights, word_in_topic_probs=None):

        self._topics_count = topics_count
        self._regularizers = regularizers
        self._regularizer_weights = regularizer_weights

        if word_in_topic_probs is not None:

            self._fixed_word_in_topic_probs = True
            self._word_in_topic_probs = word_in_topic_probs

        else:

            self._fixed_word_in_topic_probs = False

    _epsilon = 1e-10

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
        if self._fixed_word_in_topic_probs:
            word_in_topic_probs = self._word_in_topic_probs
        else:
            word_in_topic_probs = np.random.uniform(size=(words_count, self._topics_count))
            word_in_topic_probs /= word_in_topic_probs.sum(axis=0)

        # \theta_{td}
        topic_in_doc_probs = np.random.uniform(size=(self._topics_count, docs_count))
        topic_in_doc_probs /= topic_in_doc_probs.sum(axis=0)

        # n_{wt}
        word_in_topic_freqs_buffer = np.zeros_like(word_in_topic_probs)
        # n_{dt}
        topic_in_doc_freqs_buffer = np.zeros_like(topic_in_doc_probs)

        loglikelihoods = []

        for iteration_index in range(iterations_count):

            self._do_em_iteration(word_in_doc_freqs, word_in_topic_probs, topic_in_doc_probs,
                    word_in_topic_freqs_buffer, topic_in_doc_freqs_buffer)

            loglikelihood = self._get_loglikelihood(word_in_doc_freqs, word_in_topic_probs,
                        topic_in_doc_probs)

            loglikelihoods.append(loglikelihood)

            if verbose:
                print('iter#{0}: loglike={1}'.format(iteration_index + 1, loglikelihood))

        return ARTMTrainResult(word_in_doc_freqs, word_in_topic_probs, topic_in_doc_probs, words_list,
                loglikelihoods)

    def _do_em_iteration(self, word_in_doc_freqs, word_in_topic_probs, topic_in_doc_probs,
            word_in_topic_freqs_buffer, topic_in_doc_freqs_buffer):

        word_in_topics_freqs, topic_in_doc_freqs, _, _ = compute_frequencies(word_in_doc_freqs,
                word_in_topic_probs, topic_in_doc_probs, word_in_topic_freqs_buffer,
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

        if not self._fixed_word_in_topic_probs:

            np.clip(unnormalized_word_in_topic_probs, self._epsilon, None, out=unnormalized_word_in_topic_probs) 

            word_in_topic_probs[:, :] = unnormalized_word_in_topic_probs
            word_in_topic_probs /= unnormalized_word_in_topic_probs.sum(axis=0)

        np.clip(unnormalized_topic_in_doc_probs, self._epsilon, None, out=unnormalized_topic_in_doc_probs)

        topic_in_doc_probs[:, :] = unnormalized_topic_in_doc_probs
        topic_in_doc_probs /= unnormalized_topic_in_doc_probs.sum(axis=0)

    def _get_loglikelihood(self, word_in_doc_freqs, word_in_topic_probs, topic_in_doc_probs):

        loglikelihood = self.get_pure_loglikelihood(word_in_doc_freqs, word_in_topic_probs,
                topic_in_doc_probs)

        for regularizer, weight in zip(self._regularizers, self._regularizer_weights):
            loglikelihood += weight*regularizer.get_value(word_in_topic_probs, topic_in_doc_probs)

        return loglikelihood

    @classmethod
    def get_pure_loglikelihood(cls, word_in_doc_freqs, word_in_topic_probs, topic_in_doc_probs):

        loglikelihood = 0.0

        for (word_index, doc_index), word_in_doc_freq in word_in_doc_freqs.items():

            normalization_constant = word_in_topic_probs[word_index].dot(
                    topic_in_doc_probs[:, doc_index])

            loglikelihood += word_in_doc_freq*np.log(np.maximum(normalization_constant, cls._epsilon))

        return loglikelihood

class ARTMTrainResult:

    def __init__(self, word_in_doc_freqs, word_in_topic_probs, topic_in_doc_probs, words_list,
            loglikelihoods):

        self._word_in_doc_freqs = word_in_doc_freqs
        self._word_in_topic_probs = word_in_topic_probs
        self._topic_in_doc_probs = topic_in_doc_probs
        self._words_list = words_list
        self._loglikelihoods = loglikelihoods

        self._topics_count = word_in_topic_probs.shape[1]

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

        top_words_indices = self._get_top_word_indices_in_topics(top_words_count)

        return self._words_list[top_words_indices]

    def get_top_topics_in_docs(self, top_topics_count):

        return np.argsort(self._topic_in_doc_probs, axis=0)[-top_topics_count:][::-1]

    def get_train_perplexity(self):

        total_words_count = self._word_in_doc_freqs.sum()

        return np.exp(-self._loglikelihoods[-1]/total_words_count)

    def get_holdout_perplexity(self, holdout_word_in_doc_freqs, iterations_count=100, verbose=False,
            seed=None):

        holdout_freqs_part1, holdout_freqs_part2 = self._split_holdout_data(holdout_word_in_doc_freqs)

        artm = ARTM(self._topics_count, [], [], word_in_topic_probs=self._word_in_topic_probs)

        artm_part1_train_result = artm.train(holdout_freqs_part1, self._words_list,
                iterations_count=iterations_count, verbose=verbose, seed=seed)

        part1_perplexity = artm_part1_train_result.get_train_perplexity()

        part2_loglikelihood = ARTM.get_pure_loglikelihood(holdout_freqs_part2, self._word_in_topic_probs,
                artm_part1_train_result.topic_in_doc_probs)

        part2_perplexity = np.exp(-part2_loglikelihood/holdout_freqs_part2.sum())

        return part1_perplexity, part2_perplexity

    def _split_holdout_data(self, holdout_word_in_doc_freqs):

        freqs_csc = sparse.csc_matrix(holdout_word_in_doc_freqs)

        splitted_freqs_part1 = sparse.dok_matrix(holdout_word_in_doc_freqs.shape)
        splitted_freqs_part2 = sparse.dok_matrix(holdout_word_in_doc_freqs.shape)

        for doc_index in range(freqs_csc.shape[1]):

            word_indices = []

            for csc_index in range(freqs_csc.indptr[doc_index], freqs_csc.indptr[doc_index + 1]):
                word_indices += [freqs_csc.indices[csc_index]]*freqs_csc.data[csc_index]

            word_indices = np.array(word_indices)

            words_in_doc_count = len(word_indices)

            word_indices_permuted = word_indices[np.random.permutation(words_in_doc_count)]

            for element_index, word_index in enumerate(word_indices_permuted):

                if element_index < words_in_doc_count/2:
                    splitted_freqs_part1[word_index, doc_index] += 1
                else:
                    splitted_freqs_part2[word_index, doc_index] += 1

        return splitted_freqs_part1, splitted_freqs_part2

    def get_pointwise_mutual_information_metric(self, pointwise_mutual_information, top_words_count=10):

        top_word_indices_in_topics = self._get_top_word_indices_in_topics(top_words_count)

        pmi_sum = 0.0

        for topic_index in range(self._topics_count):

            for first_word_index in range(top_words_count):
                for second_word_index in range(first_word_index + 1, top_words_count):

                    smaller_index = min(first_word_index, second_word_index)
                    bigger_index = max(first_word_index, second_word_index)

                    pmi_sum += pointwise_mutual_information[smaller_index, bigger_index]

        top_pairs_count = top_words_count*(top_words_count - 1)/2

        return pmi_sum/(top_pairs_count*self._topics_count)

    def _get_top_word_indices_in_topics(self, top_words_count):

        return np.argsort(self._word_in_topic_probs, axis=0)[-top_words_count:][::-1]
