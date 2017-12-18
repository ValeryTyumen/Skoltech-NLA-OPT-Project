import opencorpora
import re
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


class OpenCorporaParser:
    """Class for parsing open corpora."""

    def __init__(self, max_features=10000, max_df=5000, min_df=20, stem=True):
        """
        :param max_features: length of the vocab
        :param stem: whether to stem words
        """
        self._stem = stem
        if stem:
            self._stemmer = SnowballStemmer("russian", ignore_stopwords=True)

        self._tf_vectorizer = CountVectorizer(max_features=max_features, max_df=max_df,
                                              min_df=min_df, stop_words='english')

    def parse_open_corpora(self, path_to_corpus='./annot.opcorpora.xml'):
        """
        Parse corpora.
        :param path_to_corpus: path to xml file
        :return: doc-term matrix: [num_docs, num_words]: csr sparse
        :return: vocab: dict of used words
        :return: year: list of year of documents
        :return: topic: list of topics
        """
        corpus = opencorpora.CorpusReader(path_to_corpus)
        documents = []
        year = []
        topic = []

        for document in corpus.iter_documents():

            document_words = self._get_document_words(document)

            if len(document_words) == 0:
                continue

            raw_sentences = " ".join(document_words) 

            documents.append(raw_sentences)
            doc_categories = document.categories()

            #process metadata
            regex_year = re.compile("Год:.*")
            doc_year = [m.group(0) for l in doc_categories
                            for m in [regex_year.search(l)] if m]

            if len(doc_year):
                year.append(int(doc_year[0].split(':')[-1]))
            else:
                year.append(-1)

            regex_topic = re.compile("Тема:.*")
            doc_topic = [m.group(0) for l in doc_categories
                             for m in [regex_topic.search(l)] if m]

            if len(doc_topic):
                topic.append(doc_topic[0].split(':')[-1].lower())
            else:
                topic.append('UNK')

        doc_term_matr = self._tf_vectorizer.fit_transform(documents)

        vocabulary = self._tf_vectorizer.vocabulary_

        close_word_pairs = self._get_close_word_pairs(corpus, vocabulary)

        return doc_term_matr, vocabulary, year, topic, close_word_pairs

    def _get_close_word_pairs(self, corpus, vocabulary):

        sliding_window_size = 10

        close_word_pairs = []

        for document in corpus.iter_documents():

            document_words = self._get_document_words(document)

            if len(document_words) == 0:
                continue

            document_close_word_pairs = set()

            for first_word_index in range(len(document_words)):

                first_word = document_words[first_word_index]

                if first_word not in vocabulary:
                    continue

                sliding_window_right_border = min(len(document_words), first_word_index + \
                        sliding_window_size)

                for second_word_index in range(first_word_index + 1, sliding_window_right_border):

                    second_word = document_words[second_word_index]

                    if second_word != first_word and second_word in vocabulary:

                        first_code = vocabulary[first_word]
                        second_code = vocabulary[second_word]

                        pair_to_add = (min(first_code, second_code), max(first_code, second_code))

                        document_close_word_pairs.add(pair_to_add)

            close_word_pairs.append(document_close_word_pairs)

        return close_word_pairs
 
    def _get_document_words(self, document):

        document_words = []

        for sentence in document.sents():
            for word in sentence:

                word = word.lower()

                if self._stem:
                    word = self._stemmer.stem(word)

                document_words.append(word)

        return document_words
