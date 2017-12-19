from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from base_regularizer import BaseRegularizer
from tools import compute_frequencies
import numpy as np

# Only one class per document


class ClassificationRegularizer(BaseRegularizer):
    """Classification regularizer."""

    def __init__(self, tau, num_topics, num_docs, docs_classes, num_words, doc_freqs):
        """
        Initialize regularizator
        :param tau: float, parameter of regularizer
        :param num_topics: int, number of topics in the model
        :param num_words: int, number of words in vocab
        :param num_docs: int, number of documents in the model
        :param docs_classes: list, number of documents, classes for documents
        """
        self._tau = tau

        self._label_encoder = LabelEncoder()
        self._encoded_classes = self._label_encoder.fit_transform(docs_classes)
        self._one_hot_encoder = OneHotEncoder()

        # m_{dc}
        self._class_in_docs = self._one_hot_encoder.fit_transform(
            self._encoded_classes.rehsape(-1, 1)).T.multiply(doc_freqs)
        self._epsilon = 1e-10

        self._num_classes = len(np.unique(self._encoded_classes))
        self._topic_in_doc_buffer = np.zeros(shape=(num_topics, num_docs))
        self._class_in_topics_buffer = np.zeros(shape=(self._num_classes, num_topics))
        self._word_in_topic_buffer = np.zeros(shape=(num_words, num_topics))

        self._class_in_docs_buffer = np.zeros(shape=(self._num_classes, num_docs))

        # psi_{ct}
        self._class_in_topics_probs = np.random.uniform(size=(self._num_classes, num_topics))

    def get_value(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Return value of regularizer.
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: float, reg value
        """
        np.dot(self._class_in_topics_probs, topic_in_doc_probs,
               out=self._class_in_docs_buffer)
        self._class_in_docs_buffer[self._class_in_docs_buffer == 0] = self._epsilon
        np.log(self._class_in_docs_buffer, out=self._class_in_docs_buffer)
        return self._tau * self._class_in_docs.multiply(self._class_in_docs_buffer).sum()

    def get_gradient(self, word_in_topic_probs, topic_in_doc_probs):
        """
        Compute gradient of regularizer w.r.t model parameters
        :param word_in_topic_probs: numpy.ndarray
        :param topic_in_doc_probs: numpy.ndarray
        :return: gradients
        """
        self._class_in_topics_probs[:, :], self._topic_in_doc_buffer[:, :], _, _ =\
            compute_frequencies(word_in_doc_freqs=self._class_in_docs,
                                word_in_topic_probs=self._class_in_topics_probs,
                                topic_in_doc_probs=topic_in_doc_probs,
                                word_in_topics_freqs_buffer=self._class_in_topics_buffer,
                                topic_in_doc_freqs_buffer=self._topic_in_doc_buffer)
        self._class_in_topics_probs /= self._class_in_topics_probs.sum(axis=0)
        self._topic_in_doc_buffer /= topic_in_doc_probs
        return self._word_in_topic_buffer, self._tau * self._topic_in_doc_buffer
