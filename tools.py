import numpy as np

def compute_frequencies(word_in_doc_freqs, word_in_topic_probs, topic_in_doc_probs,
            word_in_topics_freqs_buffer, topic_in_doc_freqs_buffer):
    """
        This logic is used inside EM-algorithm, but also in some regularizers, that's why it's here
    """

    docs_count = word_in_doc_freqs.shape[1]
    topics_count = topic_in_doc_probs.shape[0]

    word_in_topics_freqs_buffer[:, :] = 0
    # n_{wt}
    word_in_topics_freqs = word_in_topics_freqs_buffer

    topic_in_doc_freqs_buffer[:, :] = 0
    # n_{td}
    topic_in_doc_freqs = topic_in_doc_freqs_buffer

    # n_t
    topic_freqs = np.zeros(topics_count)
    # n_d
    doc_freqs = np.zeros(docs_count)

    for (word_index, doc_index), word_in_doc_freq in word_in_doc_freqs.items():

        normalization_constant = word_in_topic_probs[word_index].dot(topic_in_doc_probs[:, doc_index])

        for topic_index in range(topics_count):

            freq_increment = word_in_doc_freq*word_in_topic_probs[word_index, topic_index]*\
                    topic_in_doc_probs[topic_index, doc_index]/normalization_constant

            word_in_topics_freqs[word_index, topic_index] += freq_increment
            topic_in_doc_freqs[topic_index, doc_index] += freq_increment
            topic_freqs[topic_index] += freq_increment
            doc_freqs[doc_index] += freq_increment

    return word_in_topics_freqs, topic_in_doc_freqs, topic_freqs, doc_freqs
