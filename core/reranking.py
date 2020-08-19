import numpy as np

class Ranker:

    def __init__(self, scoring_fn, metric_type='similarity'):
        self.scoring_fn = scoring_fn
        self.metric_type = metric_type

    def score(self, query, document):
        """Calculate numerical similarity between query and document.
        
        Args:
            query (str): Text query (reference text)
            document (str): Text document
        
        Returns:
            float: Similarity score
        """
        return self.scoring_fn(query, document)

    def rank(self, query, documents):
        """Get ranks for `documents` on the basis of similarity with
            `query`.
        
        Args:
            query (str): The query (reference text)
            documents (list): Text documents
        
        Returns:
            list: Ranks for each of the documents, e.g., [2, 0, 1] means
                the document at index 0 in the input list `documents` has
                rank 2 (least similar) and document at index 1 is most
                similar. Note that the calling function has to sort the
                actual document list.
        """
        scores = [self.scoring_fn(query, document) for document in documents]
        ranks = np.argsort(scores)
        if self.metric_type == 'similarity':
            return ranks[::-1]
        else:
            return ranks

class MatchPyramidRanker(Ranker):

    def __init__(self):
        from core.matchpyramid import  calculate_similarity
        import re # needed because of an issue in MatchZoo library
        super().__init__(calculate_similarity, 'similarity')

class ConvKNRMRanker(Ranker):

    def __init__(self):
        pass

from core.representations import Text, Interaction
from core.representations import embeddings

class CustomRanker(Ranker):

    def __init__(self):
        self._interaction = Interaction()
        self._interaction.metric = 'cosine'
        self._interaction.amplify = True
        self._interaction.reinforce = False
        self._interaction.context = False
        self._interaction.window = 10
        self._interact = self._interaction.interact
        super().__init__(self.similarity, 'similarity')

    def similarity(self, query, doc):
        query_tokens = Text(query).to_tokens()
        Q = query_tokens.to_vector_sequence(embeddings)
        D = Text(doc).to_tokens().to_vector_sequence(embeddings)
        query_term_matches = self._interact(Q, D).maxpool()
        sifs = embeddings.sifs
        query_term_weights = [(sifs[word] if word in sifs else 1.0)
                                    for word in query_tokens]
        query_term_weights *= 1 - Q.redundancy_vector
        query_term_matches *= query_term_weights
        score = query_term_matches.sum()
        return score