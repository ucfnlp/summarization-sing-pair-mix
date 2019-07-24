import ssi_functions
import numpy as np
import os
from absl import flags
from absl import app
import pickle
import util
from tqdm import tqdm

FLAGS = flags.FLAGS

np.random.seed(123)

processed_data_dir = os.path.join('data/cnn_dm_processed')

def write_groundtruth_instances(f_inst, simple_similar_source_indices):
    out_str = '\t'.join(['(' + ' '.join(source_indices) + ')' for source_indices in simple_similar_source_indices]) + '\n'
    f_inst.write(out_str.encode())


def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.dataset_split == 'all':
        if FLAGS.dataset_name == 'duc_2004':
            dataset_splits = ['test']
        else:
            dataset_splits = ['test', 'val', 'train']
    else:
        dataset_splits = [FLAGS.dataset_split]
    for dataset_split in dataset_splits:

        simple_similar_source_indices_list_plus_empty = []

        processed_data_path = os.path.join(processed_data_dir, dataset_split)
        articles_path = os.path.join(processed_data_path,'articles.tsv')
        abstracts_path = os.path.join(processed_data_path,'summaries.tsv')
        highlight_path = os.path.join(processed_data_path,'highlight.tsv')
        pretty_html_path = os.path.join(processed_data_path, 'pretty_html.html')

        util.create_dirs(os.path.dirname(pretty_html_path))
        f_art = open(articles_path)
        f_abs = open(abstracts_path)
        f_hl = open(highlight_path, 'wb')
        f_pretty_html = open(pretty_html_path, 'wb')
        total = util.num_lines_in_file(articles_path)
        for example_idx in tqdm(range(total)):
            raw_article_sents = f_art.readline().strip().split('\t')
            groundtruth_summ_sents = f_abs.readline().strip().split('\t')
            article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]
            doc_indices = None
            if doc_indices is None or (FLAGS.dataset_name != 'duc_2004' and len(doc_indices) != len(
                    util.flatten_list_of_lists(article_sent_tokens))):
                doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
            doc_indices = [int(doc_idx) for doc_idx in doc_indices]
            summary_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in groundtruth_summ_sents]

            if len(article_sent_tokens) == 0:
                print('Skipping because empty')
                continue

            ''' This is the main function that finds the article sentences that were fused to create the given summary sentence'''
            simple_similar_source_indices, lcs_paths_list, smooth_article_paths_list =  ssi_functions.get_simple_source_indices_list(
                summary_sent_tokens, article_sent_tokens, vocab=None, sentence_limit=FLAGS.sentence_limit, min_matched_tokens=FLAGS.min_matched_tokens)

            simple_similar_source_indices_list_plus_empty.append(simple_similar_source_indices)
            highlight_line = '\t'.join([','.join([str(src_idx) for src_idx in source_indices]) for source_indices in simple_similar_source_indices]) + '\n'
            f_hl.write(highlight_line.encode())

            if example_idx < 5:
                extracted_sents_in_article_html = ssi_functions.html_highlight_sents_in_article(summary_sent_tokens, simple_similar_source_indices,
                                                                                  article_sent_tokens, doc_indices,
                                                                                  lcs_paths_list, smooth_article_paths_list)
                f_pretty_html.write(extracted_sents_in_article_html.encode())

        f_pretty_html.close()


if __name__ == '__main__':
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
    flags.DEFINE_string('dataset_split', 'all', 'Which dataset split to use. Must be one of {train, val, test}')
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
    flags.DEFINE_integer('top_n_sents', -1, 'Number of sentences to take from the beginning of the article. Use -1 to run on entire article.')
    flags.DEFINE_integer('min_matched_tokens', 2, 'Number of tokens required that still counts a source sentence as matching a summary sentence.')
    flags.DEFINE_integer('abstract_idx', 0, 'Which human abstract to process on. Only applies to duc_2004.')

    app.run(main)

















