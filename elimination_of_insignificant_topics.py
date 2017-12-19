from base_regularizer import BaseRegularizer
import numpy as np
from tools import compute_frequencies

class EliminationOfInsignificantTopics(BaseRegularizer):
    """Elimination of insignificant topics."""
    
    def __init__(self, tau, num_topics, num_words, num_docs, word_in_doc_freqs):
        """
        Initialize Regularizer.
        :param tau: float, parameter for regularizer
        :param num_topics: int, number of topics in the model
        :param num_words: int, vocab size
        :param num_docs: int, number of documents
        """
        self._tau = tau
        self._num_topics = num_topics
        self._num_docs = num_docs
        self._word_in_topic_buffer = np.zeros((num_words, num_topics))
        self._topic_in_doc_buffer = np.zeros((num_topics, num_docs))
        self._epsilon = 1e-10
        self._word_in_doc_freqs = word_in_doc_freqs

    def get_value(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Return value of regularizer.
        :param topic_in_doc_probs: numpy.ndarray
        :return: float, regularization value
        """

        topic_probs = topic_in_doc_probs.mean(axis=1)
        topic_probs = np.clip(topic_probs, self._epsilon, None)

        regularization_eliminate_topic = self._tau*np.log(topic_probs).sum()

        return regularization_eliminate_topic

    def get_gradient(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Compute gradient of regularizer w.r.t model parameters
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: gradient
        """

        _, _, topic_freqs, doc_freqs = compute_frequencies(self._word_in_doc_freqs, word_in_topic_probs,
                topic_in_doc_probs, self._word_in_topic_buffer, self._topic_in_doc_buffer)

        self._word_in_topic_buffer[:, :] = 0

        self._topic_in_doc_buffer[:, :] = -self._tau*doc_freqs.reshape(1, -1)
        self._topic_in_doc_buffer[:, :] /= topic_freqs.reshape(-1, 1)

        return self._word_in_topic_buffer, self._topic_in_doc_buffer
