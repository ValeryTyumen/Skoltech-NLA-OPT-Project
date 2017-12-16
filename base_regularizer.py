class BaseRegularizer:
    """
        Regularizer interface
    """

    def get_value(self, word_in_topic_probs, topic_in_doc_probs):
        """
            Returns a single value - regularizer value at the given point
        """

        raise NotImplementedError

    def get_gradient(self, word_in_topic_probs, topic_in_doc_probs):
        """
            Returns a tuple (word_in_topic_probs_grad, topic_in_doc_probs_grad)
        """

        raise NotImplementedError
