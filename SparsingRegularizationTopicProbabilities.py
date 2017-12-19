from base_regularizer import BaseRegularizer
import numpy as np

class SparsingRegularizationTopicProbabilities(BaseRegularizer):
    """Sparsing regularization of topic probabilities for the words."""
    
    def __init__(self, tau, num_topics, num_words, num_docs, word_in_doc_freq, word_in_topic_freq, topic_in_doc_freq):
        """
        Initialize Regularizer.
        :param tau: float, parameter for regularizer
        :param num_topics: int, number of topics in the model
        :param num_words: int, vocab size
        :param num_docs: int, number of documents
        :param word_in_doc_freq: sparse.dok_matrix, frequencies of words in documents
        :param word_in_topic_freq: numpy.ndarray, frequencies of words in topics
        :param topic_in_doc_freq: numpy.ndarray, frequencies of topics in documents
        """
        self._tau = tau
        self._num_topics = num_topics
        self._num_words = num_words
        self._num_docs = num_docs
        self._word_in_doc_freq = word_in_doc_freq
        self._word_in_topic_freq = word_in_topic_freq
        self._topic_in_doc_freq = topic_in_doc_freq
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
        regularization_topic_for_words_1 = \
        np.repeat((np.log(self._word_in_topic_buffer)).sum(axis = 1)[:, np.newaxis], repeats = self._num_docs,axis = 1)
        regularization_topic_for_words_2 = \
        np.repeat((np.log(self._word_in_topic_buffer)).sum(axis = 0)[np.newaxis, :], repeats = self._num_docs,axis = 0)
        regularization_topic_for_words = \
        self._tau / self._num_topics * np.multiply(self._word_in_doc_freq, 
        self._num_topics * np.log(np.dot(self._word_in_topic_buffer, self._topic_in_doc_buffer)) - 
        regularization_topic_for_words_1 - regularization_topic_for_words_2).sum()
        return regularization_topic_for_words
    
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
        self._word_in_topic_buffer[:, :] = self._tau / self._word_in_topic_buffer
        self._topic_in_doc_buffer[:, :] = self._tau / self._topic_in_doc_buffer
        partial_word_in_topic = self._word_in_topic_freq - 1 / self._num_topics * self._word_in_topic_freq.sum(axis = 1)    
        self._word_in_topic_buffer[:, :] = np.multiply(self._word_in_topic_buffer.T, partial_word_in_topic,
                                                       out=self._word_in_topic_buffer.T).T
        partial_topic_in_doc = self._topic_in_doc_freq - 1 / self._num_topics * self._topic_in_doc_freq.sum(axis = 1) 
        self._topic_in_doc_buffer[:, :] = np.multiply(self._topic_in_doc_buffer.T, b,
                                                      out=self._topic_in_doc_buffer.T).T
        return self._word_in_topic_buffer, self._topic_in_doc_buffer