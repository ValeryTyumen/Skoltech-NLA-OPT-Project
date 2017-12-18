from base_regularizer import BaseRegularizer
import numpy as np


class CovarianceDocsRegularizer(BaseRegularizer):
    """Regularizator supporting similar documents"""

    def __init__(self, tau, num_topics, num_words, num_docs, similarity_docs_matrix):
        """
        Initialize regularizator
        :param tau: float, parameter of regularizer
        :param num_topics: int, number of topics in the model
        :param num_words: int, number of words in vocab
        :param num_docs: int, number of documents in the model
        :param similarity_docs_matrix: numpy ndarray, matrix with weights of similarity, of size
        (num_docs, num_docs)
        """
        self._word_in_topic_buffer = np.zeros(shape=(num_words, num_topics))
        self._topic_in_doc_buffer = np.zeros(shape=(num_topics, num_docs))
        self._covariance_buffer = np.zeros(shape=(num_docs, num_docs))
        self._tau = tau
        self._num_topics = num_topics
        self._similarity_docs_matrix = similarity_docs_matrix

    def get_value(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Return value of regularizer.
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: float, reg value
        """
        np.dot(topic_in_doc_probs.T, topic_in_doc_probs, out=self._covariance_buffer)
        np.multiply(self._similarity_docs_matrix, self._covariance_buffer,
                    out=self._covariance_buffer)
        return self._tau * self._covariance_buffer.sum()

    def get_gradient(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Compute gradient of regularizer w.r.t model parameters
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: gradients
        """
        for i, doc in enumerate(self._similarity_docs_matrix):
            self._topic_in_doc_buffer[:, i] = np.multiply(topic_in_doc_probs,
                                                          self._similarity_docs_matrix[i]).sum(axis=1)
        return self._word_in_topic_buffer, self._tau * self._topic_in_doc_buffer
