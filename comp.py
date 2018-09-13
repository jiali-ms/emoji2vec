from gensim.models import *
import numpy as np
from sklearn.cluster import KMeans
import argparse
import os
from gensim import matutils
from util import *

# https://arxiv.org/pdf/1510.00149v5.pdf
# deep compression use k-means for quantization

parser = argparse.ArgumentParser()
parser.add_argument("--comp", "-c", type=int, default=1, help="Compression bits in range 1-32")
args = parser.parse_args()

def kmeans_compress(weight, bit=8):
    """
    compress weights using k-means

    It is a method mentioned in the paper deep compression. The input will be the original raw weights.
    Output will be compressed code like 0-255 and its centroids as its codebook.

    It takes about 1 hour to train a embedding in size like (50k, 512) with 8 CPUs cores all running.

    :param weight:
    :param bit:
    :return:
    """

    shape = weight.shape
    weight = weight.reshape(-1, 1)

    assert bit <= 32
    clusters = 2 ** bit
    kmeans = KMeans(n_clusters=clusters, n_jobs=-1)
    kmeans.fit(weight)
    code = kmeans.predict(weight)

    if bit == 8:
        code = code.astype(np.uint8)

    centroids = kmeans.cluster_centers_
    return code.reshape(shape), centroids.astype('f')

def comp_w2v():
    model = Word2Vec.load('data/word2vec.bin')
    vocab = sorted(model.wv.vocab.items(), key=lambda x: (x[1].count, x[0]), reverse=True)
    weight = np.array([model.wv[x[0]] for x in vocab])
    print('model loaded, comp bit {}'.format(args.comp))
    code, codebook = kmeans_compress(weight, args.comp)

    comp_dir = 'data/comp_{}'.format(args.comp)
    if not os.path.exists(comp_dir):
        os.makedirs(comp_dir)

    np.savetxt(os.path.join(comp_dir, 'code.txt'), code, fmt='%d')
    np.savetxt(os.path.join(comp_dir, 'codebook.txt'), codebook, fmt='%f')
    with open(os.path.join(comp_dir, 'vocab.txt'), 'w', encoding='utf-8') as f:
        for x in vocab:
            f.write('{}\t{}\n'.format(x[0], int(is_emoji(x[0]))))

def test():
    w2v = {}
    comp_dir = 'data/comp_{}'.format(args.comp)
    code = np.loadtxt(os.path.join(comp_dir, 'code.txt'), dtype=np.int8)
    codebook = np.loadtxt(os.path.join(comp_dir, 'codebook.txt'), dtype=np.float32)
    with open(os.path.join(comp_dir, 'vocab.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            w2v[word] = codebook[code[i]]

    similarity = np.inner(matutils.unitvec(w2v['💜']), matutils.unitvec(w2v['💗']))
    print('{} - {} is {}'.format('💜', '💗', similarity))

if __name__ == "__main__":
    comp_w2v()
    test()