from gensim.models import *
from util import *

class EmojiModel():
    def __init__(self):
        self.model = Word2Vec.load('data/word2vec.bin')

    def predict(self, pos, neg='', emoji_only=False):
        pos = pos.lower().strip().split()
        neg = neg.lower().strip().split()
        # print('query + {} - {}'.format(pos, neg))

        results = self.model.most_similar(positive=pos, negative=neg, topn=1000)

        if emoji_only:
            # filter out text, keep only emoji from the result
            results = [x for x in results if is_emoji(x)]

        # round the floating number for similarity
        results = [(x[0], round(x[1], 2)) for x in results]
        return results

    def similarity(self, x, y):
        print('similarity: {} - {}'.format(x, y))
        return self.model.wv.similarity(x, y)

if __name__ == "__main__":

    def show(results):
        x = ''
        for r in results:
            x += r[0] + ' {0:.2f}\t'.format(r[1])
        print(x)

    model = EmojiModel()
    #print(model.predict('king woman', 'man', emoji_only=False)[:10])
    #print(model.predict('cat', '', emoji_only=True)[:10])
    #print(model.predict('china tokyo', 'beijing', emoji_only=False)[:10])
    #print(model.predict('dog cats', 'dogs', emoji_only=False)[:5])
    #print(model.similarity('cat', 'kitten'))
    #print(model.predict('😺', emoji_only=True)[:10])
    #print(model.predict('happy new year')[:10])
    #print(model.predict('king woman', 'man')[:10])
    show(model.predict('happy', emoji_only=True)[:10])
    show(model.predict('😊', emoji_only=True)[:10])
