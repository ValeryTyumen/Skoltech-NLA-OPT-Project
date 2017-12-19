import numpy as np
from scipy import sparse

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

def get_docs_with_two_words_counts(words_count, close_word_pairs):
 
    docs_with_two_words_counts = sparse.dok_matrix((words_count, words_count), dtype=float)

    for document_close_word_pairs in close_word_pairs:
        for first_word_index, second_word_index in document_close_word_pairs:
            docs_with_two_words_counts[first_word_index, second_word_index] += 1 

    return docs_with_two_words_counts

def get_pointwise_mutual_information(word_in_doc_freqs, close_word_pairs):

    words_count = word_in_doc_freqs.shape[0]
    docs_count = word_in_doc_freqs.shape[1]

    csr_freqs = sparse.csr_matrix(word_in_doc_freqs)

    docs_with_word_counts = csr_freqs.indptr[1:] - csr_freqs.indptr[:-1]

    docs_with_two_words_counts = get_docs_with_two_words_counts(words_count, close_word_pairs)

    pointwise_mutual_information = docs_with_two_words_counts

    for first_word_index, second_word_index in list(pointwise_mutual_information.keys()):

        docs_with_two_words_count = docs_with_two_words_counts[first_word_index, second_word_index] 
        docs_with_first_word_count = docs_with_word_counts[first_word_index]
        docs_with_second_word_count = docs_with_word_counts[second_word_index]

        pointwise_mutual_information[first_word_index, second_word_index] = np.log(docs_count*\
                docs_with_two_words_count/(docs_with_first_word_count*docs_with_second_word_count))

    return pointwise_mutual_information
