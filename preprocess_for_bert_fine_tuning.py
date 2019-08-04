import itertools
import os
from tqdm import tqdm
import numpy as np
from absl import flags
from absl import app
import util
import sys
import glob
import data
import ssi_functions

FLAGS = flags.FLAGS

if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Can be {cnn_dm, xsum, duc_2004}')
if 'dataset_split' not in flags.FLAGS:
    flags.DEFINE_string('dataset_split', 'all', 'Which dataset split to use. Must be one of {train, val, test, all}')
if 'sentence_limit' not in flags.FLAGS:
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
if 'singles_and_pairs' not in flags.FLAGS:
    flags.DEFINE_string('singles_and_pairs', 'both', 'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
if 'num_instances' not in flags.FLAGS:
    flags.DEFINE_integer('num_instances', -1,
                         'Number of instances to run for before stopping. Use -1 to run on all instances.')

FLAGS(sys.argv)

from ssi_functions import filter_pairs_by_sent_position

data_dir = 'data/tf_data'
ssi_dir = 'data/ssi'
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'),
                  ('doc_indices', 'delimited_list')]
min_matched_tokens = 1
np.random.seed(123)
chronological_ssi = True

def get_bert_example(raw_article_sents, ssi):
    is_pair = len(ssi) == 2
    first_sent = raw_article_sents[ssi[0]]
    if is_pair:
        second_sent = raw_article_sents[ssi[1]]
    else:
        second_sent = ''
    return first_sent, second_sent


def get_string_bert_example(raw_article_sents, ssi, label, example_idx, inst_id):
    first_sent, second_sent = get_bert_example(raw_article_sents, ssi)
    instance = [str(label), first_sent, second_sent, str(example_idx), str(inst_id), ' '.join([str(i) for i in ssi])]
    return '\t'.join(instance) + '\n'


def main(unused_argv):

    print('Running statistics on %s' % FLAGS.dataset_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.singles_and_pairs == 'singles':
        FLAGS.sentence_limit = 1
    else:
        FLAGS.sentence_limit = 2


    if FLAGS.dataset_name == 'all':
        dataset_names = ['cnn_dm', 'xsum', 'duc_2004']
    else:
        dataset_names = [FLAGS.dataset_name]

    for dataset_name in dataset_names:
        FLAGS.dataset_name = dataset_name


        source_dir = os.path.join(data_dir, dataset_name)

        if FLAGS.dataset_split == 'all':
            if dataset_name == 'duc_2004':
                dataset_splits = ['test']
            else:
                # dataset_splits = ['val_test', 'test', 'val', 'train']
                dataset_splits = ['test', 'val', 'train']
        else:
            dataset_splits = [FLAGS.dataset_split]


        for dataset_split in dataset_splits:
            if dataset_split == 'val_test':
                source_dataset_split = 'val'
            else:
                source_dataset_split = dataset_split

            source_files = sorted(glob.glob(source_dir + '/' + source_dataset_split + '*'))

            total = len(source_files) * 1000
            example_generator = data.example_generator(source_dir + '/' + source_dataset_split + '*', True, False,
                                                       should_check_valid=False)

            out_dir = os.path.join('data', 'bert', dataset_name, FLAGS.singles_and_pairs, 'input')
            util.create_dirs(out_dir)

            writer = open(os.path.join(out_dir, dataset_split) + '.tsv', 'wb')
            header_list = ['should_merge', 'sent1', 'sent2', 'example_idx', 'inst_id', 'ssi']
            writer.write(('\t'.join(header_list) + '\n').encode())
            inst_id = 0
            for example_idx, example in enumerate(tqdm(example_generator, total=total)):
                raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, doc_indices = util.unpack_tf_example(
                    example, names_to_types)
                article_sent_tokens = [util.process_sent(sent) for sent in raw_article_sents]
                groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
                if doc_indices is None or (dataset_name != 'duc_2004' and len(doc_indices) != len(util.flatten_list_of_lists(article_sent_tokens))):
                    doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
                doc_indices = [int(doc_idx) for doc_idx in doc_indices]
                rel_sent_indices, _, _ = ssi_functions.get_rel_sent_indices(doc_indices, article_sent_tokens)
                similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, FLAGS.sentence_limit)

                possible_pairs = [x for x in
                                  list(itertools.combinations(list(range(len(raw_article_sents))), 2))]  # all pairs
                possible_pairs = filter_pairs_by_sent_position(possible_pairs, rel_sent_indices=rel_sent_indices)
                possible_singles = [(i,) for i in range(len(raw_article_sents))]
                positives = [ssi for ssi in similar_source_indices_list]

                if dataset_split == 'test' or dataset_split == 'val_test':
                    if FLAGS.singles_and_pairs == 'singles':
                        possible_combinations = possible_singles
                    else:
                        possible_combinations = possible_pairs + possible_singles
                    negatives = [ssi for ssi in possible_combinations if not (ssi in positives or ssi[::-1] in positives)]

                    for ssi_idx, ssi in enumerate(positives):
                        if len(ssi) == 0:
                            continue
                        if chronological_ssi and len(ssi) >= 2:
                            if ssi[0] > ssi[1]:
                                ssi = (min(ssi), max(ssi))
                        writer.write(get_string_bert_example(raw_article_sents, ssi, 1, example_idx, inst_id).encode())
                        inst_id += 1
                    for ssi in negatives:
                        writer.write(get_string_bert_example(raw_article_sents, ssi, 0, example_idx, inst_id).encode())
                        inst_id += 1

                else:
                    positive_sents = list(set(util.flatten_list_of_lists(positives)))
                    negative_pairs = [pair for pair in possible_pairs if not any(i in positive_sents for i in pair)]
                    negative_singles = [sing for sing in possible_singles if not sing[0] in positive_sents]
                    random_negative_pairs = np.random.permutation(len(negative_pairs)).tolist()
                    random_negative_singles = np.random.permutation(len(negative_singles)).tolist()

                    for ssi in similar_source_indices_list:
                        if len(ssi) == 0:
                            continue
                        if chronological_ssi and len(ssi) >= 2:
                            if ssi[0] > ssi[1]:
                                ssi = (min(ssi), max(ssi))
                        is_pair = len(ssi) == 2
                        writer.write(get_string_bert_example(raw_article_sents, ssi, 1, example_idx, inst_id).encode())
                        inst_id += 1

                        # False sentence single/pair
                        if is_pair:
                            if len(random_negative_pairs) == 0:
                                continue
                            negative_indices = negative_pairs[random_negative_pairs.pop()]
                        else:
                            if len(random_negative_singles) == 0:
                                continue
                            negative_indices = negative_singles[random_negative_singles.pop()]
                        article_lcs_paths = None
                        writer.write(get_string_bert_example(raw_article_sents, negative_indices, 0, example_idx, inst_id).encode())
                        inst_id += 1




if __name__ == '__main__':
    app.run(main)



