from base_regularizer import BaseRegularizer
import numpy as np


class SmoothingRegularizer(BaseRegularizer):
    """Smoothing Regularizer."""

    def __init__(self, beta_0, alpha_0, beta, alpha, num_topics, num_words, num_docs):
        """
        Initialize Regularizer.
        :param beta_0: float, parameter for word regularizer
        :param alpha_0: float, parameter for document regularizer
        :param beta: numpy.ndarray, prior Dirichlet distribution on vocab, of size (len(vocab),)
        :param alpha: numpy.ndarray, prior Dirichlet distribution on topics, of size (num_topics,)
        :param num_topics: int, number of topics in the model
        :param num_words: int, vocab size
        :param num_docs: int, number of documents
        """
        self._beta_0 = beta_0
        self._alpha_0 = alpha_0
        self._beta = beta
        self._alpha = alpha
        self._num_topics = num_topics
        self._num_words = num_words
        self._num_docs = num_docs
        self._word_in_topic_buffer = np.zeros(shape=(num_words, num_topics))
        self._topic_in_doc_buffer = np.zeros(shape=(num_topics, num_docs))
        self._epsilon = 1e-10

    def get_value(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Return value of regularizer.
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: float, reg value
        """
        np.copyto(self._word_in_topic_buffer, word_in_topic_probs)
        np.copyto(self._topic_in_doc_buffer, topic_in_doc_probs)
        self._word_in_topic_buffer[self._word_in_topic_buffer == 0] = self._epsilon
        self._topic_in_doc_buffer[self._topic_in_doc_buffer == 0] = self._epsilon
        np.log(self._word_in_topic_buffer, out=self._word_in_topic_buffer)
        np.log(self._topic_in_doc_buffer, out=self._topic_in_doc_buffer)
        regularization_word_in_topic = self._beta_0 * \
                                       np.multiply(self._word_in_topic_buffer.T, self._beta,
                                                   out=self._word_in_topic_buffer.T).sum()
        regularization_topic_in_doc = self._alpha_0 * \
                                      np.multiply(self._topic_in_doc_buffer.T, self._alpha,
                                                  out=self._topic_in_doc_buffer.T).sum()
        return regularization_topic_in_doc + regularization_word_in_topic

    def get_gradient(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Compute gradient of regularizer w.r.t model parameters
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: gradients
        """
        np.copyto(self._word_in_topic_buffer, word_in_topic_probs)
        np.copyto(self._topic_in_doc_buffer, topic_in_doc_probs)
        self._word_in_topic_buffer[self._word_in_topic_buffer == 0] = self._epsilon
        self._topic_in_doc_buffer[self._topic_in_doc_buffer == 0] = self._epsilon
        self._word_in_topic_buffer[:, :] = 1. / self._word_in_topic_buffer
        self._topic_in_doc_buffer[:, :] = 1. / self._topic_in_doc_buffer
        self._word_in_topic_buffer[:, :] = self._beta_0 * \
                                           np.multiply(self._word_in_topic_buffer.T, self._beta,
                                                       out=self._word_in_topic_buffer.T).T
        self._topic_in_doc_buffer[:, :] = self._alpha_0 * \
                                          np.multiply(self._topic_in_doc_buffer.T, self._alpha,
                                                      out=self._topic_in_doc_buffer.T).T
        return self._word_in_topic_buffer, self._topic_in_doc_buffer
