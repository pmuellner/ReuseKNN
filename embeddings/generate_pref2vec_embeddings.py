#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Information Retrieval Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import multiprocessing
import os
import random
import re
import sys
import time
from distutils.util import strtobool
from distutils.version import LooseVersion

import gensim
from gensim.models.word2vec import Word2Vec
from six import iteritems, itervalues, next

SHUFFLE_NEVER = 0
SHUFFLE_ALWAYS = 1
SHUFFLE_ONCE = 2


def repeat_simple(rating_tuple):
    return [rating_tuple[0]]


def repeat_identity(rating_tuple):
    return [rating_tuple[0]] * int(math.ceil(rating_tuple[1]))


REPEAT_FUNCTIONS = {
    'bin': repeat_simple,
    'id': repeat_identity}

MODELS = {'cbow': 0, 'sg': 1}
METHODS = {'ns': 0, 'hs': 1}

WORKERS = multiprocessing.cpu_count()


class DatasetAsSentencesIterable(object):
    """
    Iterable for a dataset where each iteration step is a list of ids.

    Can be item_based, where a word is an item and a sentences is the
    list of items rated by an user, or user_based (not item_based) where
    a word is an user and a sentence is the list of users that rated an
    item.
    """

    def __init__(self, input_file, item_based=True, shuffle=SHUFFLE_ALWAYS,
                 repeat_function=REPEAT_FUNCTIONS['id'], sort=True):
        """
        input_file: path to the file containing the ratings.
        item_based: if True every sentence are the items rated by an user. If
                    false every sentence are the users that rated an item.
        shuffle: whether to shuffle or not the output. Shuffle can be
                 performed each iteration or only on load.
        repeat_function: function that takes on tuple consisting of a pair
                         (id, rating) as input and output a list with the
                         desired sequence for such input
        sort: indicates if ratings should be ordered by timestamp. If no
              timestamp field is present or shuffle is requested this option
              is ignored and no sort is performed.
        """

        ratings = self.load_ratings(input_file, item_based)

        timestamps_missing = next(itervalues(ratings))[0][2] is None

        if not sort or timestamps_missing:
            do_sort = False
        else:
            do_sort = True

        sentences = {}
        for key, rating_list in iteritems(ratings):
            if do_sort:
                rating_list.sort(key=lambda t: t[2])

            # List of lists where each list is the id of the user/item that
            # made the rating, repeated as many times as the repeat function
            # "wants".
            repeated_words = (repeat_function((w, r))
                              for w, r, _ in rating_list)
            # Flatten the list to get the sentence.
            sentence = [word for sublist in repeated_words for word in sublist]
            if shuffle == SHUFFLE_ONCE:
                random.shuffle(sentence)
            sentences[key] = sentence

        self.shuffle = shuffle
        self.sentences = sentences

    @staticmethod
    def load_ratings(input_file, item_based=True):
        """
        Load the ratings matrix.

        If item_based it returns a mapping users -> items.
        If not item_based (user_based) it returns a mapping items -> users

        The returned dict maps keys to a list of tuples of the form
        ((user|item), rating, timestamp), where timestamps are None when
        the information is not present in the input file.
        """
        if item_based:
            idx_key = 0     # 0 = user_id field
            idx_value = 1   # 1 = item_id field
        else:
            idx_key = 1     # 1 = item_id field
            idx_value = 0   # 0 = user_id field
        with open(input_file) as f_in:
            mapping = {}
            for line in f_in:
                tokens = re.split(r'[\s,]', line)
                key = tokens[idx_key]
                value = tokens[idx_value]
                rating = float(tokens[2])
                timestamp = None if len(tokens) == 3 else tokens[3]
                try:
                    mapping[key].append((value, rating, timestamp))
                except KeyError:
                    mapping[key] = [(value, rating, timestamp)]

        return mapping

    def __iter__(self):
        sentences = list(itervalues(self.sentences))
        random.shuffle(sentences)
        for sentence in sentences:
            if self.shuffle == SHUFFLE_ALWAYS:
                random.shuffle(sentence)
            yield sentence

    def __len__(self):
        return len(self.sentences)


def train_with_parameters(
        sentences, size=100, window=5, min_count=3, hs=0, sg=0, n_iter=5,
        negative=0, workers=WORKERS, save=True, log=True, output_dir=None,
        prefix='', suffix='', save_out=False):

    filename_fields = {'dir': output_dir, 'dim': size, 'win': window,
                       'count': min_count, 'iters': n_iter, 'prefix': prefix,
                       'suffix': suffix}
    if hs:
        filename_fields['t_method'] = 'hs'
    else:
        filename_fields['t_method'] = 'ns{0}'.format(negative)

    if save or log:
        if sg:
            filename_fields['model'] = 'sg'
        else:
            filename_fields['model'] = 'cbow'

    output_filename = '{prefix}{dim}-{model}-{t_method}-w{win}-' + \
                      'c{count}-i{iters}-{suffix}'
    output_filename = output_filename.format(**filename_fields)
    output_file = os.path.join(output_dir, output_filename)

    logger = logging.getLogger('train')

    if save_out and hs == 0 and os.path.isfile(output_file + '.out_emb'):
        logger.info('Skipping %s.out_emb', output_file)
        return None
    elif (not save_out) and save and os.path.isfile(output_file +
                                                    '.embeddings'):
        logger.info('Skipping %s.embeddings', output_file)
        return None

    model = Word2Vec(None, size=size, window=window, min_count=min_count,
                     hs=hs, sg=sg, iter=n_iter, negative=negative,
                     workers=workers)
    model.build_vocab(sentences)
    if LooseVersion(gensim.__version__) <= LooseVersion('0.13.4.1'):
        model.train(sentences, total_examples=model.corpus_count,
                    report_delay=60.0)
    else:
        model.train(sentences, total_examples=model.corpus_count,
                    epochs=model.iter, start_alpha=model.alpha,
                    end_alpha=model.min_alpha, report_delay=60.0)

    if save:
        if LooseVersion(gensim.__version__) <= LooseVersion('0.13.4.1'):
            model.save_word2vec_format(output_file + '.embeddings')
        else:
            model.wv.save_word2vec_format(output_file + '.embeddings')

    if save_out and hs == 0:
        with open(output_file + '.out_emb', 'w') as f_out:
            output_embeddings = model.syn1neg
            print(output_embeddings.shape[0], output_embeddings.shape[1],
                  file=f_out)
            for word, vocab in iteritems(model.wv.vocab):
                row = model.syn1neg[vocab.index]
                print(word, ' '.join(str(val) for val in row), file=f_out)

    return model


def build_arg_parser():
    argparser = argparse.ArgumentParser(
        description='Generate embeddings.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-d', '--dimension', type=int, required=True,
                           help='Number of dimensions.', metavar='DIM')
    argparser.add_argument('-w', '--window-size', type=int, default=10,
                           help='Window size.', metavar='WIN')
    argparser.add_argument('-c', '--min_count', type=int, default=1,
                           help="""Minimun number of ratings needed for an
                           item to be included in the vocabulary.""",
                           metavar='COUNT')
    argparser.add_argument('-i', '--iters', type=int, default=10,
                           help='Number of iterations to train the model.',
                           metavar='ITERS')

    argparser.add_argument('-m', '--model', choices=MODELS.keys(),
                           default='cbow', help="""Which word2vec model to use
                           to build the embeddings""")
    argparser.add_argument('-t', '--training-method', choices=METHODS.keys(),
                           default='hs', help="""Which method to use to train
                           the word2vec model""")
    argparser.add_argument('-s', '--negative-samples', type=int, default=None,
                           help="""Amount of negative samples to take. This
                           parameter is mandatory for negative sampling and
                           ignored for hierarchical softmax""",
                           metavar='SAMPLES')

    argparser.add_argument('-o', '--output-dir', help="""Where to write the
                           files with the embeddings generated by the models.
                           """, required=True)

    argparser.add_argument('--function', choices=REPEAT_FUNCTIONS.keys(),
                           default='id', help="""Which function
                           to use when repeating items based on their
                           ratings""")
    argparser.add_argument('--shuffle', choices=[SHUFFLE_NEVER, SHUFFLE_ALWAYS,
                           SHUFFLE_ONCE], default=SHUFFLE_ALWAYS, type=int,
                           help="""Whether to shuffle the items or not before
                           feeding them to the word2vec model. 0 for not
                           shuffle, 1 for shuffle every training iteration, 2
                           for shuffle just once before training.""")

    argparser.add_argument('--datapath', help="""Use this file to load ratings
                           data to build the embeddings""", required=True)

    argparser.add_argument('--save-out-embs', default=False, type=strtobool,
                           help="""Save output embeddings. Only has effect if
                           negative sampling is used.""")

    argparser.add_argument('--workers', type=int, default=WORKERS,
                           help="""How many workers to use to build the word2vec
                           models""")

    item_or_user_group = argparser.add_mutually_exclusive_group()
    item_or_user_group.add_argument('--item-based', dest='user_based',
                                    action='store_false', help="""Build item
                                    based embeddings""")
    item_or_user_group.add_argument('--user-based', dest='user_based',
                                    action='store_true', help="""Build user
                                    based embeddings""")

    return argparser


def main(parameters):
    argparser = build_arg_parser()
    args = argparser.parse_args(parameters)

    if not os.path.isdir(args.output_dir):
        print("{0} is not a valid output directory".format(args.output_dir))
        argparser.print_usage()
        sys.exit(-1)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger()
    batch_start = time.time()

    logger.info('Loading dataset')
    sentences = DatasetAsSentencesIterable(
        args.datapath, not args.user_based,
        repeat_function=REPEAT_FUNCTIONS[args.function], shuffle=args.shuffle)

    if args.user_based:
        prefix = 'UB-'
    else:
        prefix = 'IB-'

    suffix = args.function
    if args.shuffle == SHUFFLE_NEVER:
        suffix += 't'
    elif args.shuffle == SHUFFLE_ONCE:
        suffix += '1'

    input_file = args.datapath
    fold_match = re.match(r'u([0-9]+).base', os.path.basename(input_file))
    if fold_match:
        fold = fold_match.group(1)
    else:
        fold = '1'
    suffix += '-fold'
    suffix += fold

    if args.training_method == 'hs':
        hs = 1
        negative = 0
    else:
        hs = 0
        negative = args.negative_samples

    sg = 0 if args.model == 'cbow' else 1

    train_with_parameters(
        sentences, size=args.dimension, window=args.window_size,
        min_count=args.min_count, hs=hs, sg=sg, n_iter=args.iters,
        negative=negative, workers=args.workers, save=True,
        output_dir=args.output_dir, prefix=prefix, suffix=suffix,
        save_out=args.save_out_embs)

    duration = int(time.time() - batch_start)

    seconds = duration % 60
    minutes = duration // 60

    if minutes >= 60:
        template = 'Completed in {0} hours, {1} minutes, {2} seconds'
        log_msg = template.format(minutes // 60, minutes % 60, seconds)
    else:
        template = 'Completed in {0} minutes, {1} seconds'
        log_msg = template.format(minutes, seconds)
    logger.info(log_msg)


if __name__ == '__main__':
    parameters = "-d 300 -w 50 -m cbow -t ns -s 10 --function id --shuffle 1 --user-based -i 300 --datapath ../../datasets/ml-1m/ratings_tab.csv -o ../results/embeddings/ml-1m/"
    main(parameters.split())
