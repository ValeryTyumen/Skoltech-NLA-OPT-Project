from base_regularizer import BaseRegularizer
import numpy as np


class CoherenceMaximizationRegularizer(BaseRegularizer):
    """Regularizer reducing covariance between topic-word distribution."""

    def __init__(self, tau, num_topics, num_words, num_docs, occurence_matrix):
        """
        Initialize regularizator
        :param tau: float, parameter of regularizer
        :param num_topics: int, number of topics in the model
        :param num_words: int, number of words in vocab
        :param num_docs: int, number of documents in the model
        :param occurence_matrix: csr_sparse, co-occurence matrix of words,
        of size=(num_words, num_words)
        """
        self._word_in_topic_buffer = np.zeros(shape=(num_words, num_topics))
        self._topic_in_doc_buffer = np.zeros(shape=(num_topics, num_docs))
        self._tau = tau
        self._occurence_matrix = occurence_matrix
        self._epsilon = 1e-10

    def get_value(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Return value of regularizer.
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: float, reg value
        """
        reg_topics = np.array(self._occurence_matrix.dot(word_in_topic_probs).multiply(
            word_in_topic_probs).sum(axis=0))
        reg_topics[reg_topics == 0] = self._epsilon
        np.log(reg_topics, out=reg_topics)
        return self._tau * reg_topics.sum()

    def get_gradient(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Compute gradient of regularizer w.r.t model parameters
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: gradients
        """
        reg_topics = np.array(self._occurence_matrix.dot(word_in_topic_probs).multiply(
            word_in_topic_probs).sum(axis=0))
        reg_topics[reg_topics == 0] = self._epsilon
        self._word_in_topic_buffer[:, :] = np.array(self._occurence_matrix.dot(
            word_in_topic_probs)) / reg_topics
        return 2 * self._tau * self._word_in_topic_buffer, self._topic_in_doc_buffer
