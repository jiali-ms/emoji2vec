from gensim.models import *
import itertools
import pickle

def train_model(debug=False):
    if debug:
        train = [['hello', 'word'],
                 ['how', 'are', 'you']]
    else:
        with open('data/corpus.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            train = [line.strip().split() for line in lines]

    # normally set embedding size between 10-200
    # window size is range of context you would like to include
    # a wider range windows size will let the vector learn more about topic
    model = Word2Vec(train, size=100, window=5, min_count=10, workers=4)
    model.save('data/word2vec.bin')
    model.wv.save_word2vec_format('data/word2vec.txt', binary=False)

if __name__ == "__main__":
    train_model(debug=False)