import ssi_functions
import sys
import nltk
import numpy as np
import struct
from tensorflow.core.example import example_pb2
import os
import glob
import convert_data
from absl import flags
from absl import app
import pickle
import util
from data import Vocab
from tqdm import tqdm

FLAGS = flags.FLAGS

data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref'
vocab_size = 50000
vocab_path = 'logs/vocab'

np.random.seed(123)

html_dir = 'data/highlight'
ssi_dir = 'data/ssi'
lambdamart_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi'

def write_groundtruth_instances(f_inst, simple_similar_source_indices):
    out_str = '\t'.join(['(' + ' '.join(source_indices) + ')' for source_indices in simple_similar_source_indices]) + '\n'
    f_inst.write(out_str.encode())


def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.all_actions:
        FLAGS.save_dataset = True
        FLAGS.print_output = True
        FLAGS.highlight = True

    original_dataset_name = 'xsum' if 'xsum' in FLAGS.dataset_name else 'cnn_dm' if ('cnn_dm' in FLAGS.dataset_name or 'duc_2004' in FLAGS.dataset_name) else ''
    vocab = Vocab(vocab_path + '_' + original_dataset_name, vocab_size) # create a vocabulary

    source_dir = os.path.join(data_dir, FLAGS.dataset_name + '_processed')
    util.create_dirs(html_dir)

    if FLAGS.dataset_split == 'all':
        if FLAGS.dataset_name == 'duc_2004':
            dataset_splits = ['test']
        else:
            dataset_splits = ['test', 'val', 'train']
    else:
        dataset_splits = [FLAGS.dataset_split]
    for dataset_split in dataset_splits:
        dataset_split_dir = os.path.join(source_dir, dataset_split)
        out_art = os.path.join(dataset_split_dir, 'articles')
        out_abs = os.path.join(dataset_split_dir, 'summaries')

        num_articles = util.num_lines_in_file(out_art)
        num_summaries = util.num_lines_in_file(out_abs)
        if num_articles != num_summaries:
            raise Exception('Num articles %d does not equal num summaries' % (num_articles, num_summaries))

        f_art = open(out_art)
        f_abs = open(out_abs)

        if FLAGS.highlight:
            out_inst = os.path.join(dataset_split_dir, 'highlight')
            f_highlight = open(out_inst, 'wb')

        if FLAGS.save_dataset:
            out_inst = os.path.join(dataset_split_dir, 'groundtruth_instances')
            f_inst = open(out_inst, 'wb')

        simple_similar_source_indices_list_plus_empty = []
        for example_idx in tqdm(range(num_articles)):
            article_text = f_art.readline()
            raw_article_sents = article_text.split('\t')
            article_sent_tokens = [util.process_sent(sent) for sent in raw_article_sents]

            summary_raw = f_abs.readline()
            all_summary_texts = [summ.split('\t') for summ in summary_raw.split('\t\t')]
            summary_text = all_summary_texts[0]
            summary_sent_tokens = [util.process_sent(sent) for sent in summary_text]

            if FLAGS.top_n_sents != -1:
                article_sent_tokens = article_sent_tokens[:FLAGS.top_n_sents]
                raw_article_sents = raw_article_sents[:FLAGS.top_n_sents]

            if len(article_sent_tokens) == 0:
                continue


            simple_similar_source_indices, lcs_paths_list, smooth_article_paths_list =  ssi_functions.get_simple_source_indices_list(
                summary_sent_tokens, article_sent_tokens, vocab, FLAGS.sentence_limit, FLAGS.min_matched_tokens)

            simple_similar_source_indices_list_plus_empty.append(simple_similar_source_indices)
            if FLAGS.save_dataset:
                write_groundtruth_instances(f_inst, simple_similar_source_indices)

            if FLAGS.highlight:
                extracted_sents_in_article_html = ssi_functions.html_highlight_sents_in_article(summary_sent_tokens, simple_similar_source_indices,
                                                                                  article_sent_tokens, doc_indices,
                                                                                  lcs_paths_list, smooth_article_paths_list)
                f_highlight.write(extracted_sents_in_article_html.encode())


        f_art.close()
        f_abs.close()
        if FLAGS.save_dataset:
            f_inst.close()

        if FLAGS.print_output:
            ssi_path = os.path.join(ssi_dir, FLAGS.dataset_name)
            util.create_dirs(ssi_path)
            if FLAGS.dataset_name == 'duc_2004' and FLAGS.abstract_idx != 0:
                abstract_idx_str = '_%d' % FLAGS.abstract_idx
            else:
                abstract_idx_str = ''
            with open(os.path.join(ssi_path, dataset_split + '_ssi' + abstract_idx_str + '.pkl'), 'wb') as f:
                pickle.dump(simple_similar_source_indices_list_plus_empty, f)
        if FLAGS.highlight:
            f_highlight.close()


if __name__ == '__main__':
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
    flags.DEFINE_string('dataset_split', 'all', 'Which dataset split to use. Must be one of {train, val (or dev), test}')
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
    flags.DEFINE_integer('top_n_sents', -1, 'Number of sentences to take from the beginning of the article. Use -1 to run on entire article.')
    flags.DEFINE_integer('min_matched_tokens', 2, 'Number of tokens required that still counts a source sentence as matching a summary sentence.')
    flags.DEFINE_integer('abstract_idx', 0, 'Which human abstract to process on. Only applies to duc_2004.')
    flags.DEFINE_boolean('consider_stopwords', False, 'Which human abstract to process on. Only applies to duc_2004.')
    flags.DEFINE_boolean('print_output', False, 'Whether to print and save the merged sentences and statistics.')
    flags.DEFINE_boolean('highlight', False, 'Whether to save an html file that shows the selected sentences as highlighted in the article.')
    flags.DEFINE_boolean('save_dataset', False, 'Whether to save features as a dataset that will be used to predict which sentences should be merged, using the LambdaMART system.')
    flags.DEFINE_boolean('all_actions', False, 'Which human abstract to process on. Only applies to duc_2004.')
    flags.DEFINE_boolean('tag_tokens', True, 'Whether to add token-level tags, representing whether this token is copied from the source to the summary.')

    app.run(main)

















