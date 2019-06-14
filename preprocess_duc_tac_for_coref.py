import glob
import json
import os
import struct
import time

import numpy as np
from absl import app
from absl import flags
from tqdm import tqdm

import convert_data
import data
import util
from create_corefs_dataset import get_corefs, fix_trailing_apostrophe_s, remove_irrelevant

flags.DEFINE_string('dataset_name', 'all', 'Which dataset to use. Makes a log dir based on name.\
                                                Must be one of {tac_2011, tac_2008, duc_2004, duc_tac, cnn_dm, all} or a custom dataset name')
flags.DEFINE_string('data_root', os.path.expanduser('~') + '/data/tf_data', 'Path to root directory for all datasets (already converted to TensorFlow examples).')
flags.DEFINE_string('dataset_split', 'test', 'Which dataset split to use. Must be one of {train, val, test}')
flags.DEFINE_string('coref_root', 'data/coref', 'Path to root directory for all datasets (already converted to TensorFlow examples).')
flags.DEFINE_string('mode', 'prepare', 'Whether to prepare for Stanford Corenlp coreference resolution, or to create tf_dataset including coreference resolution. Must be one of {prepare, create}')

FLAGS = flags.FLAGS


random_seed = 123
dataset_names = ['duc_2004', 'duc_2003', 'tac_2011', 'tac_2008', 'tac_2010']
names_to_types = [('raw_article_sents', 'string_list'), ('article', 'string'), ('abstract', 'string'), ('doc_indices', 'delimited_list')]


def main(unused_argv):
    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    start_time = time.time()
    np.random.seed(random_seed)
    if FLAGS.dataset_name == 'all':
        datasets = dataset_names
    else:
        datasets = [FLAGS.dataset_name]

    for dataset in datasets:
        coref_dir = os.path.join(FLAGS.coref_root, dataset)
        to_coref_dir = os.path.join(coref_dir, 'to_coref')
        corenlp_lists_dir = os.path.join(coref_dir, 'corenlp_lists')
        data_coref_dir = os.path.join(FLAGS.data_root, 'with_coref', dataset)

        util.create_dirs(to_coref_dir)
        util.create_dirs(corenlp_lists_dir)
        util.create_dirs(data_coref_dir)

        source_dir = os.path.join(FLAGS.data_root, dataset)

        if FLAGS.dataset_split == 'all':
            dataset_splits = ['test', 'val', 'train']
        else:
            dataset_splits = [FLAGS.dataset_split]
        for dataset_split in dataset_splits:
            source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))

            total = len(source_files) * 1000 if ('cnn' in dataset or 'newsroom' in dataset or 'xsum' in dataset) else len(source_files)
            example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False,
                                                   should_check_valid=False)

            if FLAGS.mode == 'prepare':
                corenlp_list = []
                out_idx = 0
                for example_idx, example in enumerate(tqdm(example_generator, total=total)):
                    raw_article_sents, article, abstract, doc_indices = util.unpack_tf_example(example, names_to_types)
                    if raw_article_sents is None:
                        continue
                    raw_article = ' '.join(raw_article_sents)
                    file_name = os.path.join(to_coref_dir, '%s_%06d.bin'%(dataset_split, out_idx))
                    with open(file_name, 'wb') as f:
                        f.write(raw_article)
                    corenlp_list.append(file_name)
                with open(os.path.join(corenlp_lists_dir, 'all_' + dataset_split + '.txt'), 'wb') as f:
                    f.write('\n'.join(corenlp_list))
                    out_idx += 1


            elif FLAGS.mode == 'create':
                process_coref_dir = os.path.join(coref_dir, 'processed')

                out_idx = 0
                out_file_name = os.path.join(data_coref_dir, dataset_split + '_{:05d}.bin'.format(out_idx // 1000))
                writer = open(os.path.join(out_file_name), 'wb')
                coref_files = sorted(glob.glob(os.path.join(process_coref_dir, dataset_split + '*')))
                coref_dict = {}
                for c in coref_files:
                    coref_dict[c.split('/')[-1].split('.json')[0]] = c
                print(len(coref_files), len(source_files))
                for example_idx, example in enumerate(tqdm(example_generator, total=total)):
                    raw_article_sents, article, abstract, doc_indices = util.unpack_tf_example(example, names_to_types)
                    if raw_article_sents is None:
                        continue
                    raw_article_sents = [sent for sent in raw_article_sents if sent.strip() != '']
                    if out_idx % 1000 == 0 and out_idx != 0:
                        writer.close()
                        out_file_name = os.path.join(data_coref_dir, dataset_split + '_{:05d}.bin'.format(out_idx // 1000))
                        writer = open(os.path.join(out_file_name), 'wb')

                    # coref_file = os.path.join(process_coref_dir, 'test_%06d.bin.json' % example_idx)
                    # coref_file = coref_files[out_idx]
                    # matched_files = [name for name in coref_files if '%s_%06d.bin'%(dataset_split, out_idx) in name]
                    file_name = '%s_%06d.bin'%(dataset_split, out_idx)
                    if file_name in coref_dict:
                        file_path = coref_dict[file_name]
                        corefs = get_corefs(file_path)
                        fixed_corefs = fix_trailing_apostrophe_s(corefs)

                        corefs_relevant_info = remove_irrelevant(fixed_corefs)
                        corefs_json = json.dumps(corefs_relevant_info)
                    else:
                        corefs_json = json.dumps([])

                    example.features.feature['corefs'].bytes_list.value.extend([corefs_json])

                    tf_example = convert_data.make_example(article, abstract, doc_indices, raw_article_sents, corefs)

                    convert_data.write_tf_example(example, writer)

                    out_idx += 1
                writer.close()


                    # file_name = os.path.join(data_coref_dir, '%s_%06d.bin' % (dataset_split, example_idx))
                    # writer = open(file_name, 'wb')
                    # coref_file = os.path.join(process_coref_dir, 'test_%06d.bin.json'%example_idx)
                    # corefs = get_corefs(coref_file)
                    # fixed_corefs = fix_trailing_apostrophe_s(corefs)
                    #
                    # corefs_relevant_info = remove_irrelevant(fixed_corefs)
                    # corefs_json = json.dumps(corefs_relevant_info)
                    #
                    # example.features.feature['corefs'].bytes_list.value.extend([corefs_json])
                    # tf_example_str = example.SerializeToString()
                    # str_len = len(tf_example_str)
                    # writer.write(struct.pack('q', str_len))
                    # writer.write(struct.pack('%ds' % str_len, tf_example_str))
                    #
                    # writer.close()


    util.print_execution_time(start_time)

















if __name__ == '__main__':
    app.run(main)














