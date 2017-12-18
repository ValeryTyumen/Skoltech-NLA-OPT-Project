from base_regularizer import BaseRegularizer
from smoothing_regularizer import SmoothingRegularizer
import numpy as np


class SparsingRegularizer(BaseRegularizer):
    """Sparsing regularization for the model."""

    def __init__(self,  beta_0, alpha_0, beta, alpha, num_topics, num_words, num_docs):
        self._smoothing_reg = SmoothingRegularizer(beta_0=-beta_0, alpha_0=-alpha_0, beta=beta,
                                                  alpha=alpha, num_topics=num_topics,
                                                  num_words=num_words, num_docs=num_docs)

    def get_value(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Return value of regularizer.
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: float, reg value
        """
        return self._smoothing_reg.get_value(word_in_topic_probs=word_in_topic_probs,
                                            topic_in_doc_probs=topic_in_doc_probs)

    def get_gradient(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Compute gradient of regularizer w.r.t model parameters
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: gradients
        """
        return self._smoothing_reg.get_gradient(word_in_topic_probs=word_in_topic_probs,
                                               topic_in_doc_probs=topic_in_doc_probs)