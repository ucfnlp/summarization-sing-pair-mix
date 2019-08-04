from tqdm import tqdm
import rouge_functions
from absl import flags
from absl import app
import time
import glob
import numpy as np
import data
import os
import sys
import util
from ssi_functions import html_highlight_sents_in_article, get_simple_source_indices_list
import pickle
import ssi_functions
import multiprocessing as mp

if 'dataset_name' in flags.FLAGS:
    flags_already_done = True
else:
    flags_already_done = False

FLAGS = flags.FLAGS
if 'singles_and_pairs' not in flags.FLAGS:
    flags.DEFINE_string('singles_and_pairs', 'both', 'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', '')
if 'upper_bound' not in flags.FLAGS:
    flags.DEFINE_boolean('upper_bound', False, 'If true, then uses the groundtruth singletons/pairs.')
if 'num_instances' not in flags.FLAGS:
    flags.DEFINE_integer('num_instances', -1, 'Number of instances to run for before stopping. Use -1 to run on all instances.')
if 'sentemb' not in flags.FLAGS:
    flags.DEFINE_boolean('sentemb', True, 'Adds sentence position embedding to every word in BERT.')
if 'artemb' not in flags.FLAGS:
    flags.DEFINE_boolean('artemb', True, 'Adds arrticle embedding that is used when giving a score to a given instance in BERT.')
if 'plushidden' not in flags.FLAGS:
    flags.DEFINE_boolean('plushidden', True, 'Adds an extra hidden layer at the output layer of BERT.')
# flags.DEFINE_boolean('l_sents', True, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')

if not flags_already_done:
    FLAGS(sys.argv)

_exp_name = 'bert'
dataset_split = 'test'
filter_sentences = True
random_seed = 123
max_sent_len_feat = 20
min_matched_tokens = 2

data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi'
bert_in_dir = os.path.join('data', 'bert', FLAGS.dataset_name, FLAGS.singles_and_pairs, 'input')
bert_scores_dir = os.path.join('data', 'bert', FLAGS.dataset_name, FLAGS.singles_and_pairs, 'output')
ssi_out_dir = 'data/temp/' + FLAGS.dataset_name + '/ssi'
log_dir = 'logs'
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'), ('doc_indices', 'delimited_list')]

if FLAGS.singles_and_pairs == 'both':
    exp_name = FLAGS.dataset_name + '_' + _exp_name + '_both'
    dataset_articles = FLAGS.dataset_name
else:
    exp_name = FLAGS.dataset_name + '_' + _exp_name + '_singles'
    dataset_articles = FLAGS.dataset_name + '_singles'

if FLAGS.upper_bound:
    exp_name = exp_name + '_upperbound'

if FLAGS.singles_and_pairs == 'singles':
    sentence_limit = 1
else:
    sentence_limit = 2

if FLAGS.dataset_name == 'xsum':
    l_param = 40
else:
    l_param = 100
# l_param = 100
temp_in_path = os.path.join(bert_in_dir, 'test.tsv')
temp_out_path = os.path.join(bert_scores_dir, 'test_results.tsv')
util.create_dirs(bert_scores_dir)
my_log_dir = os.path.join(log_dir, exp_name)
dec_dir = os.path.join(my_log_dir, 'decoded')
ref_dir = os.path.join(my_log_dir, 'reference')
html_dir = os.path.join(my_log_dir, 'hightlighted_html')
util.create_dirs(dec_dir)
util.create_dirs(ref_dir)
util.create_dirs(html_dir)
util.create_dirs(ssi_out_dir)

def read_bert_scores(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    data = [[float(x) for x in line.split('\t')] for line in lines]
    data = np.array(data)
    return data

def get_qid_source_indices(line):
    items = line.split('\t')
    qid = int(items[3])
    inst_id = int(items[4])
    source_indices = [int(x) for x in items[5].split()]

    if len(source_indices) == 2:
        source_indices = [min(source_indices), max(source_indices)]

    return qid, inst_id, source_indices

def read_source_indices_from_bert_input(file_path):
    out_list = []
    with open(file_path) as f:
        lines = f.readlines()[1:]
    for line in lines:
        qid, inst_id, source_indices = get_qid_source_indices(line)
        out_list.append(tuple((qid, tuple(source_indices))))
    return out_list

def get_sent_or_sents(article_sent_tokens, source_indices):
    chosen_sent_tokens = [article_sent_tokens[idx] for idx in source_indices]
    # sents = util.flatten_list_of_lists(chosen_sent_tokens)
    return chosen_sent_tokens

def get_bert_scores_for_singles_pairs(data, qid_source_indices_list):
    out_dict = {}
    for row_idx, row in enumerate(data):
        score0, score1 = row
        qid, source_indices = qid_source_indices_list[row_idx]
        if qid not in out_dict:
            out_dict[qid] = {}
        out_dict[qid][source_indices] = score1
    return out_dict

def rank_source_sents(temp_in_path, temp_out_path):
    qid_source_indices_list = read_source_indices_from_bert_input(temp_in_path)
    data = read_bert_scores(temp_out_path)
    if len(qid_source_indices_list) != len(data):
        raise Exception('Len of qid_source_indices_list %d != Len of data %d' % (len(qid_source_indices_list), len(data)))
    source_indices_to_scores = get_bert_scores_for_singles_pairs(data, qid_source_indices_list)
    return source_indices_to_scores

def get_best_source_sents(article_sent_tokens, mmr_dict, already_used_source_indices):
    if len(already_used_source_indices) == 0:
        source_indices = max(mmr_dict, key=mmr_dict.get)
    else:
        best_value = -9999999
        best_source_indices = ()
        for key, val in mmr_dict.items():
            if val > best_value and not any(i in list(key) for i in already_used_source_indices):
                best_value = val
                best_source_indices = key
        source_indices = best_source_indices
    sents = get_sent_or_sents(article_sent_tokens, source_indices)
    return sents, source_indices

def generate_summary(article_sent_tokens, qid_ssi_to_importances, example_idx):
    qid = example_idx

    summary_sent_tokens = []
    summary_tokens = util.flatten_list_of_lists(summary_sent_tokens)
    already_used_source_indices = []
    similar_source_indices_list = []
    summary_sents_for_html = []
    ssi_length_extractive = None

    # Iteratively select a singleton/pair from the article that has the highest score from BERT
    while len(summary_tokens) < 300:
        if len(summary_tokens) >= l_param and ssi_length_extractive is None:
            ssi_length_extractive = len(similar_source_indices_list)
        mmr_dict = util.calc_MMR_source_indices(article_sent_tokens, summary_tokens, None, qid_ssi_to_importances, qid=qid)
        sents, source_indices = get_best_source_sents(article_sent_tokens, mmr_dict, already_used_source_indices)
        if len(source_indices) == 0:
            break
        summary_sent_tokens.extend(sents)
        summary_tokens = util.flatten_list_of_lists(summary_sent_tokens)
        similar_source_indices_list.append(source_indices)
        summary_sents_for_html.append(' <br> '.join([' '.join(sent) for sent in sents]))
        if filter_sentences:
            already_used_source_indices.extend(source_indices)
    if ssi_length_extractive is None:
        ssi_length_extractive = len(similar_source_indices_list)
    selected_article_sent_indices = util.flatten_list_of_lists(similar_source_indices_list[:ssi_length_extractive])
    summary_sents = [' '.join(sent) for sent in util.reorder(article_sent_tokens, selected_article_sent_indices)]
    return summary_sents, similar_source_indices_list, summary_sents_for_html, ssi_length_extractive

def example_generator_extended(example_generator, total, single_feat_len, pair_feat_len, singles_and_pairs):
    example_idx = -1
    for example in tqdm(example_generator, total=total):
        example_idx += 1
        if FLAGS.num_instances != -1 and example_idx >= FLAGS.num_instances:
            break
        yield (example, example_idx, single_feat_len, pair_feat_len, singles_and_pairs)

def evaluate_example(ex):
    example, example_idx, qid_ssi_to_importances, _, _ = ex
    print(example_idx)

    # Read example from dataset
    raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, doc_indices = util.unpack_tf_example(example, names_to_types)
    article_sent_tokens = [util.process_sent(sent) for sent in raw_article_sents]
    enforced_groundtruth_ssi_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, sentence_limit)
    groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
    groundtruth_summ_sent_tokens = [sent.split(' ') for sent in groundtruth_summ_sents[0]]

    if FLAGS.upper_bound:
        # If upper bound, then get the groundtruth singletons/pairs
        replaced_ssi_list = util.replace_empty_ssis(enforced_groundtruth_ssi_list, raw_article_sents)
        selected_article_sent_indices = util.flatten_list_of_lists(replaced_ssi_list)
        summary_sents = [' '.join(sent) for sent in util.reorder(article_sent_tokens, selected_article_sent_indices)]
        similar_source_indices_list = groundtruth_similar_source_indices_list
        ssi_length_extractive = len(similar_source_indices_list)
    else:
        # Generates summary based on BERT output. This is an extractive summary.
        summary_sents, similar_source_indices_list, summary_sents_for_html, ssi_length_extractive = generate_summary(article_sent_tokens, qid_ssi_to_importances, example_idx)
        similar_source_indices_list_trunc = similar_source_indices_list[:ssi_length_extractive]
        summary_sents_for_html_trunc = summary_sents_for_html[:ssi_length_extractive]
        if example_idx <= 100:
            summary_sent_tokens = [sent.split(' ') for sent in summary_sents_for_html_trunc]
            extracted_sents_in_article_html = html_highlight_sents_in_article(summary_sent_tokens, similar_source_indices_list_trunc,
                                            article_sent_tokens, doc_indices=doc_indices)

            groundtruth_ssi_list, lcs_paths_list, article_lcs_paths_list = get_simple_source_indices_list(
                                            groundtruth_summ_sent_tokens,
                                           article_sent_tokens, None, sentence_limit, min_matched_tokens)
            groundtruth_highlighted_html = html_highlight_sents_in_article(groundtruth_summ_sent_tokens, groundtruth_ssi_list,
                                            article_sent_tokens, lcs_paths_list=lcs_paths_list, article_lcs_paths_list=article_lcs_paths_list, doc_indices=doc_indices)

            all_html = '<u>System Summary</u><br><br>' + extracted_sents_in_article_html + '<u>Groundtruth Summary</u><br><br>' + groundtruth_highlighted_html
            ssi_functions.write_highlighted_html(all_html, html_dir, example_idx)
    rouge_functions.write_for_rouge(groundtruth_summ_sents, summary_sents, example_idx, ref_dir, dec_dir)
    return (groundtruth_similar_source_indices_list, similar_source_indices_list, ssi_length_extractive)


def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    print('Running statistics on %s' % exp_name)

    start_time = time.time()
    np.random.seed(random_seed)
    source_dir = os.path.join(data_dir, dataset_articles)
    source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))


    total = len(source_files)*1000
    example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False, should_check_valid=False)

    # Read output of BERT and put into a dictionary with:
    # key=(article idx, source indices {this is a tuple of length 1 or 2, depending on if it is a singleton or pair})
    # value=score
    qid_ssi_to_importances = rank_source_sents(temp_in_path, temp_out_path)
    ex_gen = example_generator_extended(example_generator, total, qid_ssi_to_importances, None, FLAGS.singles_and_pairs)
    print('Creating list')
    ex_list = [ex for ex in ex_gen]

    # Main function to get results on all test examples
    pool = mp.Pool(mp.cpu_count())
    ssi_list = pool.map(evaluate_example, ex_list)
    pool.close()

    # ssi_list = list(map(evaluate_example, ex_list))

    # save ssi_list
    with open(os.path.join(my_log_dir, 'ssi.pkl'), 'wb') as f:
        pickle.dump(ssi_list, f)
    with open(os.path.join(my_log_dir, 'ssi.pkl'), 'rb') as f:
        ssi_list = pickle.load(f)
    print('Evaluating BERT model F1 score...')
    suffix = util.all_sent_selection_eval(ssi_list)
    print('Evaluating ROUGE...')
    results_dict = rouge_functions.rouge_eval(ref_dir, dec_dir, l_param=l_param)
    rouge_functions.rouge_log(results_dict, my_log_dir, suffix=suffix)

    ssis_restricted = [ssi_triple[1][:ssi_triple[2]] for ssi_triple in ssi_list]
    ssi_lens = [len(source_indices) for source_indices in util.flatten_list_of_lists(ssis_restricted)]
    num_singles = ssi_lens.count(1)
    num_pairs = ssi_lens.count(2)
    print ('Percent singles/pairs: %.2f %.2f' % (num_singles*100./len(ssi_lens), num_pairs*100./len(ssi_lens)))

    util.print_execution_time(start_time)


if __name__ == '__main__':
    # main()
    app.run(main)




































