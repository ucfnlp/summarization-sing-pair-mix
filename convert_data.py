# -*- coding: utf8 -*-

"""
Script to convert multi-document inputs to TensorFlow examples which can be sent to the PG-MMR model.
"""

import struct
from tensorflow.core.example import example_pb2
import nltk
import os
from absl import flags
from absl import app
from tqdm import tqdm
import json
import util
import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf8')
except:
    _=None

FLAGS = flags.FLAGS

p_start_tag = '<P>'
p_end_tag = '</P>'


def convert_singpairmix_to_tf_examples(dataset_name, processed_data_dir, tf_example_dir, dataset_split='all'):
    out_dir = os.path.join(tf_example_dir, dataset_name)
    out_full_dir = os.path.join(out_dir, 'all')
    util.create_dirs(out_full_dir)
    if dataset_split == 'all':
        if dataset_name == 'duc_2004':
            dataset_splits = ['test']
        else:
            dataset_splits = ['test', 'val', 'train']
    else:
        dataset_splits = [dataset_split]
    for dataset_split in dataset_splits:
        processed_data_path = os.path.join(processed_data_dir, dataset_name, dataset_split)
        articles_path = os.path.join(processed_data_path,'articles.tsv')
        abstracts_path = os.path.join(processed_data_path,'summaries.tsv')
        highlight_path = os.path.join(processed_data_path,'highlight.tsv')

        f_art = open(articles_path)
        f_abs = open(abstracts_path)
        f_hl = open(highlight_path)
        writer = open(os.path.join(out_full_dir, dataset_split + '.bin'), 'wb')
        total = util.num_lines_in_file(articles_path)
        for example_idx in tqdm(range(total)):
            raw_article_sents = f_art.readline().strip().split('\t')
            groundtruth_summ_sents = f_abs.readline().strip().split('\t')
            summary_text = '\n'.join(groundtruth_summ_sents)
            article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]
            doc_indices = None
            if doc_indices is None or (dataset_name != 'duc_2004' and len(doc_indices) != len(
                    util.flatten_list_of_lists(article_sent_tokens))):
                doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
            similar_source_indices = [source_indices.split(',') for source_indices in f_hl.readline().split('\t')]

            write_bert_tf_example(similar_source_indices, raw_article_sents, summary_text, None,
                                     None, None, writer, dataset_name)

        writer.close()
        if dataset_name == 'cnn_dm' or dataset_name == 'newsroom' or dataset_name == 'xsum':
            chunk_size = 1000
        else:
            chunk_size = 1
        util.chunk_file(dataset_split, out_full_dir, out_dir, chunk_size=chunk_size)

def write_bert_tf_example(simple_similar_source_indices, raw_article_sents, summary_text, corefs_str, doc_indices, article_lcs_paths_list, writer, dataset_name):
    tf_example = example_pb2.Example()
    source_indices_str = ';'.join([' '.join(str(i) for i in source_indices) for source_indices in simple_similar_source_indices])
    tf_example.features.feature['similar_source_indices'].bytes_list.value.extend([util.encode_text(source_indices_str)])
    for sent in raw_article_sents:
        s = sent.strip()
        tf_example.features.feature['raw_article_sents'].bytes_list.value.extend([util.encode_text(s)])
    if dataset_name == 'duc_2004':
        for summ_text in summary_text:
            tf_example.features.feature['summary_text'].bytes_list.value.extend([util.encode_text(summ_text)])
    else:
        tf_example.features.feature['summary_text'].bytes_list.value.extend([util.encode_text(summary_text)])
    if doc_indices is not None:
        tf_example.features.feature['doc_indices'].bytes_list.value.extend([util.encode_text(doc_indices)])
    if corefs_str is not None:
        tf_example.features.feature['corefs'].bytes_list.value.extend([corefs_str])
    if article_lcs_paths_list is not None:
        article_lcs_paths_list_str = '|'.join([';'.join([' '.join(str(i) for i in source_indices) for source_indices in article_lcs_paths]) for article_lcs_paths in article_lcs_paths_list])
        tf_example.features.feature['article_lcs_paths_list'].bytes_list.value.extend([util.encode_text(article_lcs_paths_list_str)])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))

def process_abstract(abstract_lines):
    abstract = ''
    for line in abstract_lines:
        line = line.lower()
        line = line.replace('\x92', "'")
        tokenized_sent = nltk.word_tokenize(line)
        tokenized_sent = [util.fix_bracket_token(token) for token in tokenized_sent]
        sent = ' '.join(tokenized_sent)
        abstract += '<s> ' + sent + ' </s> '
    abstract = abstract.strip()
    return abstract

def make_example(article, abstracts, doc_indices, raw_article_sents, corefs, article_lcs_paths=None):
    tf_example = example_pb2.Example()
    tf_example.features.feature['article'].bytes_list.value.extend([util.encode_text(article)])
    for abstract in abstracts:
        if type(abstract) == list:
            tf_example.features.feature['abstract'].bytes_list.value.extend([util.encode_text(process_abstract(abstract))])
        else:
            tf_example.features.feature['abstract'].bytes_list.value.extend([util.encode_text(abstract)])
    if doc_indices is not None:
        if type(doc_indices) == list:
            doc_indices = ' '.join(doc_indices)
        tf_example.features.feature['doc_indices'].bytes_list.value.extend([util.encode_text(doc_indices)])
    if raw_article_sents is not None:
        for sent in raw_article_sents:
            tf_example.features.feature['raw_article_sents'].bytes_list.value.extend([util.encode_text(sent)])
    if corefs is not None:
        corefs_str = json.dumps(corefs)
        tf_example.features.feature['corefs'].bytes_list.value.extend([util.encode_text(corefs_str)])
    if article_lcs_paths is not None:
        article_lcs_paths_str = ';'.join([' '.join(str(i) for i in source_indices) for source_indices in article_lcs_paths])
        tf_example.features.feature['article_lcs_paths'].bytes_list.value.extend([util.encode_text(article_lcs_paths_str)])
    return tf_example

def write_example(article, abstracts, doc_indices, raw_article_sents, corefs, writer):
    tf_example = make_example(article, abstracts, doc_indices, raw_article_sents, corefs)
    write_tf_example(tf_example, writer)

def write_tf_example(tf_example, writer):
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))

def write_with_generator(example_generator, num_examples, out_dir, data_split):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_idx = 1
    out_file_name = os.path.join(out_dir, data_split + '_{:05d}.bin'.format(out_idx // 1000 + 1))
    writer = open(os.path.join(out_file_name), 'wb')
    for example in tqdm(example_generator, total=num_examples):
        if (out_idx - 1) % 1000 == 0 and out_idx != 1:
            writer.close()
            out_file_name = os.path.join(out_dir, data_split + '_{:05d}.bin'.format(out_idx // 1000 + 1))
            writer = open(os.path.join(out_file_name), 'wb')
        write_tf_example(example, writer)

        out_idx += 1
    writer.close()
    a = 0

def main(unused_argv):
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    if FLAGS.dataset_name == '':
        raise Exception('Must specify which dataset to convert.')
    convert_singpairmix_to_tf_examples(FLAGS.dataset_name, FLAGS.line_by_line_data_path, FLAGS.out_data_path, dataset_split=FLAGS.dataset_split)


if __name__ == '__main__':
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to convert from raw data to tf examples')
    flags.DEFINE_string('dataset_split', 'all', 'Which dataset split to use. Must be one of {train, val, test, all}')
    flags.DEFINE_string('line_by_line_data_path', 'data/processed', 'Where the data is, to be converted to TF examples.')
    flags.DEFINE_string('out_data_path', 'data/tf_data', 'Where to put output tf examples')
    app.run(main)
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    