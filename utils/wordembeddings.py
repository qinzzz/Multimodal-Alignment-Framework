from scipy import spatial
import numpy as np


class WordEmbeddings():
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
            word_idx = self.word_indexer.index_of(word)
            if word_idx != -1:
                return self.vectors[word_idx]
            else:
                return self.vectors[self.word_indexer.index_of("UNK")]

    def get_embeddings(self, word_list):
            emb_list = []
            for word in word_list:
                emb = self.get_embedding(word)
                emb_list.append(emb)
            return np.array(emb_list)

    def similarity(self, w1, w2):
        return 1 - spatial.distance.cosine(self.get_embedding(w1), self.get_embedding(w2))

