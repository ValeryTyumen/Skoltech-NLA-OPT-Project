from base_regularizer import BaseRegularizer
from smoothing_regularizer import SmoothingRegularizer
from sparsing_regularizer import SparsingRegularizer
import numpy as np


class CombinedSmoothingSparsingRegularizer(BaseRegularizer):
    """Combining smoothing and sparsing regularization for the model."""

    def __init__(self, beta_0, alpha_0, beta, alpha, num_topics, num_words, num_docs,
                 domain_specific_topics, background_topics):
        """
        Initialize Regularizer.
        :param beta_0: float, parameter for word regularizer
        :param alpha_0: float, parameter for document regularizer
        :param beta: numpy.ndarray, prior Dirichlet distribution on vocab, of size (len(vocab),)
        :param alpha: numpy.ndarray, prior Dirichlet distribution on topics, of size (num_topics,)
        :param num_topics: int, number of topics in the model
        :param num_words: int, vocab size
        :param num_docs: int, number of documents
        :param domain_specific_topics: indexes of topics considered domain specific
        :param background_topics: indexes of topics considered background topics
        """
        self._sparsing_reg = SparsingRegularizer(beta_0=beta_0, alpha_0=alpha_0, beta=beta,
                                                 alpha=alpha[domain_specific_topics],
                                                 num_topics=len(domain_specific_topics),
                                                 num_words=num_words, num_docs=num_docs)
        self._smoothing_reg = SmoothingRegularizer(beta_0=beta_0, alpha_0=alpha_0, beta=beta,
                                                   alpha=alpha[background_topics],
                                                   num_topics=len(background_topics),
                                                   num_words=num_words, num_docs=num_docs)
        self._word_in_topic_buffer = np.zeros(shape=(num_words, num_topics))
        self._topic_in_doc_buffer = np.zeros(shape=(num_topics, num_docs))
        self._domain_specific_topics = domain_specific_topics
        self._background_topics = background_topics

    def get_value(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Return value of regularizer.
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: float, reg value
        """
        reg_specific_value = self._sparsing_reg.get_value(
            word_in_topic_probs=word_in_topic_probs[:, self._domain_specific_topics],
            topic_in_doc_probs=topic_in_doc_probs[self._domain_specific_topics])
        reg_background_value = self._smoothing_reg.get_value(
            word_in_topic_probs=word_in_topic_probs[:, self._background_topics],
            topic_in_doc_probs=topic_in_doc_probs[self._background_topics])
        return reg_specific_value + reg_background_value

    def get_gradient(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Compute gradient of regularizer w.r.t model parameters
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: gradients
        """
        self._word_in_topic_buffer[:, self._domain_specific_topics], \
        self._topic_in_doc_buffer[self._domain_specific_topics] = \
            self._sparsing_reg.get_gradient(
                word_in_topic_probs=word_in_topic_probs[:, self._domain_specific_topics],
                topic_in_doc_probs=topic_in_doc_probs[self._domain_specific_topics])
        self._word_in_topic_buffer[:, self._background_topics], \
        self._topic_in_doc_buffer[self._background_topics] = \
            self._smoothing_reg.get_gradient(
                word_in_topic_probs=word_in_topic_probs[:, self._background_topics],
                topic_in_doc_probs=topic_in_doc_probs[self._background_topics])
        return self._word_in_topic_buffer, self._topic_in_doc_buffer
