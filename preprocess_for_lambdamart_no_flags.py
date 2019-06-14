import scipy
import time
import itertools
import convert_data
import numpy as np
import data
from tqdm import tqdm
import util
from absl import flags
from absl import app
import sys
import os
import hashlib
import struct
import subprocess
import collections
import glob
from tensorflow.core.example import example_pb2
from scipy import sparse
from scoop import futures
from collections import defaultdict
import pickle
# from multiprocessing.dummy import Pool as ThreadPool
# pool = ThreadPool(12)

if 'singles_and_pairs' in flags.FLAGS:
    flags_already_done = True
else:
    flags_already_done = False
FLAGS = flags.FLAGS
if 'singles_and_pairs' not in flags.FLAGS:
    flags.DEFINE_string('singles_and_pairs', 'singles', 'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
if 'dataset_split' not in flags.FLAGS:
    flags.DEFINE_string('dataset_split', 'train_val', 'Which dataset split to use. Must be one of {train, val, test}')
if 'use_pair_criteria' not in flags.FLAGS:
    flags.DEFINE_boolean('use_pair_criteria', False, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'pca' not in flags.FLAGS:
    flags.DEFINE_boolean('pca', False, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'tfidf_limit' not in flags.FLAGS:
    flags.DEFINE_integer('tfidf_limit', -1, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'num_instances' not in flags.FLAGS:
    flags.DEFINE_integer('num_instances', -1, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'sent_position_criteria' not in flags.FLAGS:
    flags.DEFINE_boolean('sent_position_criteria', True, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'special_xsum_balance' not in flags.FLAGS:
    flags.DEFINE_boolean('special_xsum_balance', True, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'lr' not in flags.FLAGS:
    flags.DEFINE_boolean('lr', False, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')

if not flags_already_done:
    FLAGS(sys.argv)

exp_name = 'reference'
num_instances = -1,
random_seed = 123
max_sent_len_feat = 20
balance = True
importance = True
real_values = True
# singles_and_pairs = 'singles'
include_sents_dist = True
include_tfidf_vec = True
min_matched_tokens = 1

data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi'
log_dir = 'logs/'
out_dir = 'data/to_lambdamart'
tfidf_vec_path = 'data/tfidf/' + 'all' + '_tfidf_vec_5.pkl'
pca_vec_path = 'data/tfidf/' + 'all' + '_pca.pkl'
temp_dir = 'data/temp'
max_enc_steps = 100000
min_dec_steps = 100
max_dec_steps = 120

dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_lists'), ('summary_text', 'string'), ('corefs', 'json'), ('doc_indices', 'delimited_list')]

print('Loading TFIDF vectorizer')
with open(tfidf_vec_path, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

if FLAGS.pca:
    print('Loading LSA model')
    with open(pca_vec_path, 'rb') as f:
        pca = pickle.load(f)
else:
    pca = None

def convert_to_one_hot(value, bins, range):
    hist, _ = np.histogram(value, bins=bins, range=range)
    return hist.tolist()

def does_start_with_quotation_mark(sent_tokens):
    if len(sent_tokens) == 0:
        return False
    return sent_tokens[0] == "`" or sent_tokens[0] == "``"

max_num_sents = 30
def get_single_sent_features(sent_idx, sent_term_matrix, article_sent_tokens, mmr, rel_sent_idx):
    abs_sent_idx = rel_sent_idx + 1.0
    norm_sent_idx = (rel_sent_idx + 1.0) / max_num_sents        # POSSIBLY A BUG, NEED TO DO MIN(REL_SENT_IDX, MAX_NUM_SENTS)
    # doc_similarity = util.cosine_similarity(sent_term_matrix[sent_idx], doc_vector)[0][0]
    sent_len = len(article_sent_tokens[sent_idx])
    sent_len = min(max_sent_len_feat, sent_len)
    starts_with_quote = int(does_start_with_quotation_mark(article_sent_tokens[sent_idx])) + 1
    my_mmr = mmr[sent_idx]
    if scipy.sparse.issparse(sent_term_matrix):
        tfidf_vec = sent_term_matrix[sent_idx].toarray()[0].tolist()
    else:
        tfidf_vec = sent_term_matrix[sent_idx].tolist()

    if real_values:
        features = [abs_sent_idx, norm_sent_idx, sent_len, starts_with_quote, my_mmr]
        if include_tfidf_vec:
            features.extend(tfidf_vec)
        return features
    else:
        sent_idx, _ = np.histogram(min(sent_idx, max_num_sents), bins=10, range=(0,max_num_sents))
        # doc_similarity, _ = np.histogram(doc_similarity, bins=5, range=(0,1))
        sent_len, _ = np.histogram(sent_len, bins=10, range=(1,max_sent_len_feat))
        my_mmr = convert_to_one_hot(my_mmr, 5, (0,1))
        return sent_idx.tolist() + sent_len.tolist() + [starts_with_quote] + my_mmr

def get_pair_sent_features(similar_source_indices, sent_term_matrix, article_sent_tokens, mmr, my_rel_sent_indices):
    features = []
    # features.append(1)  # is_sent_pair
    sent_idx1, sent_idx2 = similar_source_indices[0], similar_source_indices[1]
    rel_sent_idx1, rel_sent_idx2 = my_rel_sent_indices[0], my_rel_sent_indices[1]
    sent1_features = get_single_sent_features(sent_idx1,
                         sent_term_matrix, article_sent_tokens, mmr, rel_sent_idx1)
    features.extend(sent1_features) # sent_idx, doc_similarity, sent_len
    sent2_features = get_single_sent_features(sent_idx2,
                         sent_term_matrix, article_sent_tokens, mmr, rel_sent_idx2)
    features.extend(sent2_features) # sent_idx, doc_similarity, sent_len
    average_mmr = (mmr[sent_idx1] + mmr[sent_idx2])/2
    sent1_row = sent_term_matrix[sent_idx1]
    sent2_row = sent_term_matrix[sent_idx2]
    if FLAGS.pca:
        sent1_row = sent1_row.reshape(1, -1)
        sent2_row = sent2_row.reshape(1, -1)
    sents_similarity = util.cosine_similarity(sent1_row, sent2_row)[0][0]
    sents_dist = abs(rel_sent_idx1 - rel_sent_idx2)
    if real_values:
        features.extend([average_mmr, sents_similarity])
        if include_sents_dist:
            features.append(sents_dist)
    else:
        features.extend(convert_to_one_hot(average_mmr, 5, (0,1)))
        features.extend(convert_to_one_hot(sents_similarity, 5, (0,1))) # sents_similarity
        if include_sents_dist:
            features.extend(convert_to_one_hot(min(sents_dist, max_num_sents), 10, (0,max_num_sents))) # sents_dist
    return features


def get_features(similar_source_indices, sent_term_matrix, article_sent_tokens, rel_sent_indices, single_feat_len,
                 pair_feat_len, mmr, singles_and_pairs):
    features = []
    if len(similar_source_indices) == 1:
        if singles_and_pairs == 'pairs':
            return None
        sent_idx = similar_source_indices[0]
        rel_sent_idx = rel_sent_indices[sent_idx]
        features = get_single_sent_features(sent_idx, sent_term_matrix, article_sent_tokens, mmr, rel_sent_idx)
        if singles_and_pairs == 'both':
            features = [2] + features
            features.extend([0]*pair_feat_len)
    elif len(similar_source_indices) == 2:
        if singles_and_pairs == 'singles':
            return None
        if singles_and_pairs == 'both':
            features = [1] + features
            features.extend([0]*single_feat_len)
        my_rel_sent_indices = [rel_sent_indices[similar_source_indices[0]], rel_sent_indices[similar_source_indices[1]]]
        features.extend(get_pair_sent_features(similar_source_indices, sent_term_matrix, article_sent_tokens, mmr, my_rel_sent_indices))
    elif len(similar_source_indices) == 0:
        return None
    else:
        print(similar_source_indices)
        raise Exception("Shouldn't be here")
    return features


def format_to_lambdamart(inst, single_feat_len):
    features, relevance, query_id, source_indices, inst_id = inst.features, inst.relevance, inst.qid, inst.source_indices, inst.inst_id
    if query_id == 0:
        a=0
    if features is None or len(features) == 0:
        raise Exception('features has no elements')
    is_single_sent = features[0]
    out_str = str(relevance) + ' qid:' + str(query_id)

    for feat_idx, feat in enumerate(features):
        # if singles_and_pairs == 'singles' or singles_and_pairs == 'pairs' or feat_idx == 0 or \
        #         (is_single_sent and feat_idx < single_feat_len) or (not is_single_sent and feat_idx >= single_feat_len):
        if feat != 0 or feat_idx==len(features)-1:
            out_str += ' %d:%f' % (feat_idx+1, feat)
        # else:
        #     out_str += ' %d:%f' % (feat_idx + 1, -100)

    # for feat_idx, feat in enumerate(features):
    #     if feat != 0 or feat_idx == len(features)-1:
    #         out_str += ' %d:%f' % (feat_idx+1, feat)
    out_str += ' #source_indices:'
    for idx, source_idx in enumerate(source_indices):
        out_str += str(source_idx)
        if idx != len(source_indices) - 1:
            out_str += ' '
    out_str += ',inst_id:' + str(inst_id)
    return out_str

class Lambdamart_Instance:
    def __init__(self, features, relevance, qid, source_indices):
        self.features = features
        self.relevance = relevance
        self.qid = qid
        self.source_indices = source_indices
        self.inst_id = -1

def assign_inst_ids(instances):
    qid_cur_inst_id = defaultdict(int)
    for instance in instances:
        instance.inst_id = qid_cur_inst_id[instance.qid]
        qid_cur_inst_id[instance.qid] += 1

def sentences_have_overlap(article_sent_tokens, s1, s2, min_matched_tokens):
    nonstopword_matches, _ = util.matching_unigrams(article_sent_tokens[s1], article_sent_tokens[s2], should_remove_stop_words=True)
    if len(nonstopword_matches) >= min_matched_tokens:
        return True
    else:
        return False

def filter_by_overlap(article_sent_tokens, possible_pairs):
    new_possible_pairs = []
    for s1, s2 in possible_pairs:
        if sentences_have_overlap(article_sent_tokens, s1, s2, min_matched_tokens):
            new_possible_pairs.append((s1, s2))
    return new_possible_pairs

def get_coref_pairs(corefs):
    coref_pairs = set()
    for coref in corefs:
        sent_indices = set()
        for m in coref:
            sent_idx = m['sentNum'] - 1
            sent_indices.add(sent_idx)
        pairs = list(itertools.combinations(sorted(list(sent_indices)), 2))
        coref_pairs = coref_pairs.union(pairs)
    return coref_pairs

def filter_by_entites(article_sent_tokens, possible_pairs, corefs):
    coref_pairs = get_coref_pairs(corefs)
    new_possible_pairs = coref_pairs.intersection(set(possible_pairs))
    return new_possible_pairs

def convert_article_to_lambdamart_features(ex):
    # example_idx += 1
    # if num_instances != -1 and example_idx >= num_instances:
    #     break
    example, example_idx, single_feat_len, pair_feat_len, singles_and_pairs, out_path = ex
    print(example_idx)
    raw_article_sents, similar_source_indices_list, summary_text, corefs, doc_indices = util.unpack_tf_example(example, names_to_types)
    article_sent_tokens = [util.process_sent(sent) for sent in raw_article_sents]
    if doc_indices is None:
        doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
    doc_indices = [int(doc_idx) for doc_idx in doc_indices]
    if len(doc_indices) != len(util.flatten_list_of_lists(article_sent_tokens)):
        doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
    rel_sent_indices, _, _ = get_rel_sent_indices(doc_indices, article_sent_tokens)
    if FLAGS.singles_and_pairs == 'singles':
        sentence_limit = 1
    else:
        sentence_limit = 2
    similar_source_indices_list = util.enforce_sentence_limit(similar_source_indices_list, sentence_limit)
    summ_sent_tokens = [sent.strip().split() for sent in summary_text.strip().split('\n')]

    # sent_term_matrix = util.get_tfidf_matrix(raw_article_sents)
    article_text = ' '.join(raw_article_sents)
    sent_term_matrix = util.get_doc_substituted_tfidf_matrix(tfidf_vectorizer, raw_article_sents, article_text, pca)
    doc_vector = np.mean(sent_term_matrix, axis=0)

    out_str = ''
    # ssi_idx_cur_inst_id = defaultdict(int)
    instances = []

    if importance:
        importances = util.special_squash(util.get_tfidf_importances(tfidf_vectorizer, raw_article_sents, pca))
        possible_pairs = [x for x in list(itertools.combinations(list(range(len(raw_article_sents))), 2))]   # all pairs
        if FLAGS.use_pair_criteria:
            possible_pairs = filter_pairs_by_criteria(raw_article_sents, possible_pairs, corefs)
        if FLAGS.sent_position_criteria:
            possible_pairs = filter_pairs_by_sent_position(possible_pairs, rel_sent_indices)
        possible_singles = [(i,) for i in range(len(raw_article_sents))]
        possible_combinations = possible_pairs + possible_singles
        positives = [ssi for ssi in similar_source_indices_list]
        negatives = [ssi for ssi in possible_combinations if not (ssi in positives or ssi[::-1] in positives)]

        negative_pairs = [x for x in possible_pairs if not (x in similar_source_indices_list or x[::-1] in similar_source_indices_list)]
        negative_singles = [x for x in possible_singles if not (x in similar_source_indices_list or x[::-1] in similar_source_indices_list)]
        random_negative_pairs = np.random.permutation(len(negative_pairs)).tolist()
        random_negative_singles = np.random.permutation(len(negative_singles)).tolist()

        qid = example_idx
        for similar_source_indices in positives:
            # True sentence single/pair
            relevance = 1
            features = get_features(similar_source_indices, sent_term_matrix, article_sent_tokens, rel_sent_indices, single_feat_len, pair_feat_len, importances, singles_and_pairs)
            if features is None:
                continue
            instances.append(Lambdamart_Instance(features, relevance, qid, similar_source_indices))
            a=0

            if FLAGS.dataset_name == 'xsum' and FLAGS.special_xsum_balance:
                neg_relevance = 0
                num_negative = 4
                if FLAGS.singles_and_pairs == 'singles':
                    num_neg_singles = num_negative
                    num_neg_pairs = 0
                else:
                    num_neg_singles = num_negative/2
                    num_neg_pairs = num_negative/2
                for _ in range(num_neg_singles):
                    if len(random_negative_singles) == 0:
                        continue
                    negative_indices = negative_singles[random_negative_singles.pop()]
                    neg_features = get_features(negative_indices, sent_term_matrix, article_sent_tokens, rel_sent_indices, single_feat_len, pair_feat_len, importances, singles_and_pairs)
                    if neg_features is None:
                        continue
                    instances.append(Lambdamart_Instance(neg_features, neg_relevance, qid, negative_indices))
                for _ in range(num_neg_pairs):
                    if len(random_negative_pairs) == 0:
                        continue
                    negative_indices = negative_pairs[random_negative_pairs.pop()]
                    neg_features = get_features(negative_indices, sent_term_matrix, article_sent_tokens, rel_sent_indices, single_feat_len, pair_feat_len, importances, singles_and_pairs)
                    if neg_features is None:
                        continue
                    instances.append(Lambdamart_Instance(neg_features, neg_relevance, qid, negative_indices))
            elif balance:
                # False sentence single/pair
                is_pair = len(similar_source_indices) == 2
                if is_pair:
                    if len(random_negative_pairs) == 0:
                        continue
                    negative_indices = negative_pairs[random_negative_pairs.pop()]
                else:
                    if len(random_negative_singles) == 0:
                        continue
                    negative_indices = negative_singles[random_negative_singles.pop()]
                neg_relevance = 0
                neg_features = get_features(negative_indices, sent_term_matrix, article_sent_tokens, rel_sent_indices, single_feat_len, pair_feat_len, importances, singles_and_pairs)
                if neg_features is None:
                    continue
                instances.append(Lambdamart_Instance(neg_features, neg_relevance, qid, negative_indices))
        if not balance:
            for negative_indices in negatives:
                neg_relevance = 0
                neg_features = get_features(negative_indices, sent_term_matrix, article_sent_tokens, single_feat_len, pair_feat_len, importances, singles_and_pairs)
                if neg_features is None:
                    continue
                instances.append(Lambdamart_Instance(neg_features, neg_relevance, qid, negative_indices))

    sorted_instances = sorted(instances, key=lambda x: (x.qid, x.source_indices))
    assign_inst_ids(sorted_instances)
    if FLAGS.lr:
        return sorted_instances
    else:
        for instance in sorted_instances:
            lambdamart_str = format_to_lambdamart(instance, single_feat_len)
            out_str += lambdamart_str + '\n'
        with open(os.path.join(out_path, '%06d.txt' % example_idx), 'wb') as f:
            f.write(out_str)
        # print out_str
        # return out_str

def example_generator_extended(example_generator, total, single_feat_len, pair_feat_len, singles_and_pairs, out_path):
    example_idx = -1
    for example in tqdm(example_generator, total=total):
    # for example in example_generator:
        example_idx += 1
        if num_instances != -1 and example_idx >= num_instances:
            break
        yield (example, example_idx, single_feat_len, pair_feat_len, singles_and_pairs, out_path)

# ####Delete all flags before declare#####
#
# def del_all_flags(FLAGS):
#     flags_dict = _flags()
#     keys_list = [keys for keys in flags_dict]
#     for keys in keys_list:
#         __delattr__(keys)

# del_all_flags(FLAGS)
def main(unused_argv):
    print('Running statistics on %s' % exp_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.singles_and_pairs == 'both':
        in_dataset = FLAGS.dataset_name
        out_dataset = FLAGS.dataset_name + '_both'
    else:
        in_dataset = FLAGS.dataset_name + '_singles'
        out_dataset = FLAGS.dataset_name + '_singles'

    if FLAGS.lr:
        out_dataset = FLAGS.dataset_name + '_lr'

    start_time = time.time()
    np.random.seed(random_seed)
    source_dir = os.path.join(data_dir, in_dataset)
    ex_sents = ['single .', 'sentence .']
    article_text = ' '.join(ex_sents)
    sent_term_matrix = util.get_doc_substituted_tfidf_matrix(tfidf_vectorizer, ex_sents, article_text, pca)
    if FLAGS.singles_and_pairs == 'pairs':
        single_feat_len = 0
    else:
        single_feat_len = len(get_single_sent_features(0, sent_term_matrix, [['single','.'],['sentence','.']], [0,0], 0))
    if FLAGS.singles_and_pairs == 'singles':
        pair_feat_len = 0
    else:
        pair_feat_len = len(get_pair_sent_features([0,1], sent_term_matrix, [['single','.'],['sentence','.']], [0,0], [0, 0]))
    util.print_vars(single_feat_len, pair_feat_len)
    util.create_dirs(temp_dir)

    if FLAGS.dataset_split == 'all':
        dataset_splits = ['test', 'val', 'train']
    elif FLAGS.dataset_split == 'train_val':
        dataset_splits = ['val', 'train']
    else:
        dataset_splits = [FLAGS.dataset_split]
    for split in dataset_splits:
        source_files = sorted(glob.glob(source_dir + '/' + split + '*'))

        out_path = os.path.join(out_dir, out_dataset, split)
        if FLAGS.pca:
            out_path += '_pca'
        util.create_dirs(os.path.join(out_path))
        total = len(source_files)*1000 if ('cnn' in in_dataset or 'newsroom' in in_dataset or 'xsum' in in_dataset) else len(source_files)
        example_generator = data.example_generator(source_dir + '/' + split + '*', True, False, should_check_valid=False)
        # for example in tqdm(example_generator, total=total):
        ex_gen = example_generator_extended(example_generator, total, single_feat_len, pair_feat_len, FLAGS.singles_and_pairs, out_path)
        print('Creating list')
        ex_list = [ex for ex in ex_gen]
        if FLAGS.num_instances != -1:
            ex_list = ex_list[:FLAGS.num_instances]
        print('Converting...')
        # all_features = pool.map(convert_article_to_lambdamart_features, ex_list)



        # all_features = ray.get([convert_article_to_lambdamart_features.remote(ex) for ex in ex_list])


        if FLAGS.lr:
            all_instances = list(futures.map(convert_article_to_lambdamart_features, ex_list))
            all_instances = util.flatten_list_of_lists(all_instances)
            x = [inst.features for inst in all_instances]
            x = np.array(x)
            y = [inst.relevance for inst in all_instances]
            y = np.expand_dims(np.array(y), 1)
            x_y = np.concatenate((x, y), 1)
            np.save(writer, x_y)
        else:
            list(futures.map(convert_article_to_lambdamart_features, ex_list))
            # writer.write(''.join(all_features))

        # all_features = []
        # for example  in tqdm(ex_gen, total=total):
        #     all_features.append(convert_article_to_lambdamart_features(example))

        # all_features = util.flatten_list_of_lists(all_features)
        # num1 = sum(x == 1 for x in all_features)
        # num2 = sum(x == 2 for x in all_features)
        # print 'Single sent: %d instances. Pair sent: %d instances.' % (num1, num2)

        # for example in tqdm(ex_gen, total=total):
        #     features = convert_article_to_lambdamart_features(example)
        #     writer.write(features)

        final_out_path = out_path + '.txt'
        file_names = sorted(glob.glob(os.path.join(out_path, '*')))
        writer = open(final_out_path, 'wb')
        for file_name in tqdm(file_names):
            with open(file_name) as f:
                text = f.read()
            writer.write(text)
        writer.close()
    util.print_execution_time(start_time)


if __name__ == '__main__':

    app.run(main)






























