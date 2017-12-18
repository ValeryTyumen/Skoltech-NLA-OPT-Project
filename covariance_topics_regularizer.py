from base_regularizer import BaseRegularizer
import numpy as np


class CovarianceTopicsRegularizer(BaseRegularizer):
    """Regularizator reducing covariance between topic-word distribution."""

    def __init__(self, tau, num_topics, num_words, num_docs):
        """
        Initialize regularizator
        :param tau: float, parameter of regularizer
        :param num_topics: int, number of topics in the model
        :param num_words: int, number of words in vocab
        :param num_docs: int, number of documents in the model
        """
        self._word_in_topic_buffer = np.zeros(shape=(num_words, num_topics))
        self._topic_in_doc_buffer = np.zeros(shape=(num_topics, num_docs))
        self._covariance_buffer = np.zeros(shape=(num_topics, num_topics))
        self._tau = tau
        self._num_topics = num_topics

    def get_value(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Return value of regularizer.
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: float, reg value
        """
        np.dot(word_in_topic_probs.T, word_in_topic_probs, out=self._covariance_buffer)
        np.fill_diagonal(self._covariance_buffer, 0.0)
        return -self._tau * self._covariance_buffer.sum()

    def get_gradient(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Compute gradient of regularizer w.r.t model parameters
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: gradients
        """
        self._word_in_topic_buffer[:, :] = np.repeat(word_in_topic_probs.sum(axis=1)[:, np.newaxis],
                                                     repeats=self._num_topics, axis=1)
        self._word_in_topic_buffer -= word_in_topic_probs
        return -self._tau * self._word_in_topic_buffer, self._topic_in_doc_buffer
