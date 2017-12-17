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
            raw_sentences = document.sents()
            if len(raw_sentences):
                if self._stem:
                    raw_sentences = " ".join([" ".join([self._stemmer.stem(word.lower())
                                                        for word in sentence])
                                              for sentence in raw_sentences])
                else:
                    raw_sentences = " ".join([" ".join([word.lower() for word in sentence])
                                              for sentence in raw_sentences])

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
        return doc_term_matr, self._tf_vectorizer.vocabulary_, year, topic
