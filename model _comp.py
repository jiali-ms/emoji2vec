import numpy as np
from sklearn.preprocessing import normalize
import os
from util import *

class EmojiModelCompressed():
    def __init__(self, comp=4):
        self.comp = comp
        comp_dir = 'data/comp_{}'.format(self.comp)
        self.w2i = {}
        self.i2w = {}
        self.code = np.loadtxt(os.path.join(comp_dir, 'code.txt'), dtype=np.int8)
        self.codebook = np.loadtxt(os.path.join(comp_dir, 'codebook.txt'), dtype=np.float32)
        self.vectors = normalize(self.codebook[self.code], axis=1, norm='l2')
        with open(os.path.join(comp_dir, 'vocab.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                word = line.strip().split('\t')[0]
                self.i2w[i] = word
                self.w2i[word] = i

    def _most_similar(self, w, topn=1000):
        w_v = self.vectors[self.w2i[w]]
        scores = np.dot(self.vectors, w_v)
        most_similar = np.argsort(scores)[::-1][:topn]
        return [(self.i2w[x], scores[x]) for x in most_similar if not x == self.w2i[w]]

    def predict(self, w, emoji_only=False):
        results = self._most_similar(w)

        if emoji_only:
            # filter out text, keep only emoji from the result
            results = [x for x in results if is_emoji(x)]

        # round the floating number for similarity
        results = [(x[0], round(x[1], 2)) for x in results]
        return results

    def similarity(self, x, y):
        print('similarity: {} - {}'.format(x, y))
        return np.inner(self.vectors[self.w2i[x]], self.vectors[self.w2i[y]])

if __name__ == "__main__":
    model = EmojiModelCompressed()

    def show(results):
        x = ''
        for r in results:
            x += r[0] + ' {0:.2f}\t'.format(r[1])
        print(x)

    #print(model.similarity('dog', 'cat'))
    #print(model.predict('cat', emoji_only=True))
    show(model.predict('happy', emoji_only=True)[:10])
    show(model.predict('😊', emoji_only=True)[:10])