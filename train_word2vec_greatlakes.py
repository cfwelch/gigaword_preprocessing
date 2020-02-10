

import unicodedata, logging, random, nltk, json, gc, sys, os

from tqdm import tqdm
from gensim.models import Word2Vec, FastText
from gensim.models.callbacks import CallbackAny2Vec
from argparse import ArgumentParser

class EpochLogger(CallbackAny2Vec):
    def __init__(self, dimension, min_count, embed_root):
        self.epoch = 0
        self.last_loss = 0.0
        self.dimension = dimension
        self.min_count = min_count
        self.embed_root = embed_root
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_diff = loss - self.last_loss
        self.last_loss = loss
        print('Epoch ' + str(self.epoch) + ' ending loss: ' + '{:.5f}'.format(loss_diff))
        self.epoch += 1
        if not os.path.isdir(self.embed_root):
            os.makedirs(self.embed_root)
        model.save(self.embed_root + '/word2vec_mc' + str(self.min_count) + '_e' + str(self.epoch) + '_d' + str(self.dimension) + '.model')
        model.wv.save_word2vec_format(self.embed_root + '/word2vec_mc' + str(self.min_count) + '_e' + str(self.epoch) + '_d' + str(self.dimension) + '_text')

def main():
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', dest='epochs', help='Number of epochs to train for. Defaults to 10.', default=10, type=int)
    parser.add_argument('-w', '--workers', dest='workers', help='Number of threads to use. Defaults to 4.', default=4, type=int)
    parser.add_argument('-d', '--dimension', dest='dimension', help='Number of dimensions of the word vectors. Defaults to 100.', default=100, type=int)
    parser.add_argument('-mc', '--min_count', dest='min_count', help='Minimum count for words to be included in the vocabulary. Defaults to 3.', default=3, type=int)
    # parser.add_argument('-s', '--subsample', dest='subsample', help='Sample sentences from the input file instead of using all of them. Defaults to using all of them.', default=-1, type=int)
    # parser.add_argument('-f', '--filename', dest='filename', default=None, help='Name of the sentences file. Required.', type=str)
    parser.add_argument('-er', '--embed_root', dest='embed_root', help='Embedding root directory for saving files. Defaults to \'embeddings\'.', type=str)
    opt = parser.parse_args()

    all_sentences = []
    file_names = ['flat_gigaword_1_preproc', 'flat_gigaword_2_preproc', 'flat_gigaword_3_preproc']
    for file_name in file_names:
        print('Reading file: ' + file_name)
        with open(file_name) as handle:
            for line in tqdm(handle):
                all_sentences.append(line.strip().split(' '))

    # Do word2vec things with gensim
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    epoch_logger = EpochLogger(opt.dimension, opt.min_count, opt.embed_root)

    print('Using word embedding method: word2vec')
    model = Word2Vec(all_sentences, size=opt.dimension, window=5, min_count=opt.min_count, workers=opt.workers, sg=1, negative=0, hs=1, iter=opt.epochs, sorted_vocab=1, compute_loss=True, callbacks=[epoch_logger])

if __name__ == '__main__':
    main()
