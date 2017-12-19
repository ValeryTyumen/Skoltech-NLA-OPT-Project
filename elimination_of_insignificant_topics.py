from base_regularizer import BaseRegularizer
import numpy as np

class EliminationOfInsignificantTopics(BaseRegularizer):
    """Elimination of insignificant topics."""
    
    def __init__(self, tau, num_docs, word_in_topic_freq, topic_in_doc_freq):
        """
        Initialize Regularizer.
        :param tau: float, parameter for regularizer
        :param num_docs: int, number of documents
        :param word_in_topic_freq: numpy.ndarray, frequencies of words in topics
        :param topic_in_doc_freq: numpy.ndarray, frequencies of topics in documents
        """
        self._tau = tau
        self._num_topics = num_topics
        self._num_docs = num_docs
        self._word_in_topic_freq = word_in_topic_freq
        self._topic_in_doc_freq = topic_in_doc_freq
        self._topic_in_doc_buffer = np.zeros(shape=(num_topics, num_docs))
        self._epsilon = 1e-10
        
    def get_value(self, topic_in_doc_probs):
        """
        Return value of regularizer.
        :param topic_in_doc_probs: numpy.ndarray
        :return: float, regularization value
        """
        
        np.copyto(self._topic_in_doc_buffer, topic_in_doc_probs)
        self._topic_in_doc_buffer[self._topic_in_doc_buffer == 0] = self._epsilon
        np.multiply(self._topic_in_doc_buffer, 1 / self._num_docs, out=self._topic_in_doc_buffer).sum(axis = 1)
        regularization_eliminate_topic = tau * np.log(self._topic_in_doc_buffer).sum()
        return regularization_eliminate_topic
    
    def get_gradient(self, topic_in_doc_probs):
        """
        Compute gradient of regularizer w.r.t model parameters
        :param topic_in_doc_probs: numpy.ndarray
        :return: gradient
        """
        np.copyto(self._topic_in_doc_buffer, topic_in_doc_probs)
        self._topic_in_doc_buffer[self._topic_in_doc_buffer == 0] = self._epsilon
        self._topic_in_doc_buffer[:, :] = \
        self._tau * self._topic_in_doc_freq.sum(axis = 1) / self._word_in_topic_freq.sum()
        return self._topic_in_doc_buffer