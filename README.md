# Skoltech-NLA-OPT-Project
ARTM implementation project for assignment at Skoltech uni

## Subtasks to split between team members (you can edit the README to assign yourself):

1. Dataset parsing. You need to parse OpenCorpora dataset with and without lemmas (in the latter case you have to put the data through Mystem using some Python wrapperor on hand). You have to extract metadata as well in a convenient form. **Sasha** wanted to do that AFAIK.

2. EM algorithm for ARTM. **I** feel like doing that myself.

3. Implementing regularizers as subclasses of some unified "Regularizer" oracle with methods for regularizer function and its derivatives. Be aware that not all of them might be used in the result report.

	* Smoothing regularization.

	* Sparsing regularization.

	* Combining smoothing and sparsing.

	* Semi-supervised learning (??).

	* Sparsing regularization of topic probabilities for the words.

	* Elimination of insignificant topics.

	* Covariance regularization for topics.

	* Covariance regularization for documents.

	* Coherence maximization.

	* The classification regularizer (use meta-data as classes).

	* Label regularization.

4. Implementing quality metrics (perplexity, sparsity, interpetability, coherence) and hold-out testing. As Lempitsky said, we should choose one metric as the main one to compare different models. I feel like coherence might be the choice, correct me if I'm wrong.

5. Plotting and visual evaluation of topics.

6. Final report, presentation. **Marina** wanted to prepare a report.
