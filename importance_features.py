import numpy as np
import itertools
import util, data
from sklearn.metrics.pairwise import cosine_similarity
import batcher
from absl import flags
from sklearn import svm
import glob
import tqdm
import pickle

FLAGS = flags.FLAGS

class SentRep:
    def __init__(self, abs_sent_indices, rel_sent_indices_0_to_10, sent_lens,
        sent_representations_separate, cluster_rep_sent_separate, dist_separate):
        self.abs_sent_indices = abs_sent_indices
        self.rel_sent_indices_0_to_10 = rel_sent_indices_0_to_10
        self.sent_lens = sent_lens
        self.sent_representations_separate = sent_representations_separate
        self.cluster_rep_sent_separate = cluster_rep_sent_separate
        self.dist_separate = dist_separate
        self.y = None
        self.binary_y = None

def get_features_list(include_y):
    features = []
    features.append('abs_sent_indices')
    features.append('rel_sent_indices_0_to_10')
    features.append('sent_lens')
    features.append('sent_representations_separate')
    features.append('cluster_rep_sent_separate')
    features.append('dist_separate')
    if include_y:
        if FLAGS.importance_fn == 'svr':
            features.append('y')
        else:
            features.append('binary_y')

    return features

def get_importance_features_for_article(enc_states, enc_sentences, sent_indices, tokenizer, sent_representations_separate):
    abs_sent_indices = sent_indices
    rel_sent_indices_0_to_10 = get_relative_sent_indices(sent_indices)
    sent_lens = get_sent_lens(enc_sentences)
    cluster_rep_sent_separate, dist_separate = get_cluster_representations(sent_representations_separate)
    assert len(sent_lens) == len(sent_representations_separate)

    sent_reps = []
    for i in range(len(abs_sent_indices)):
        sent_reps.append(SentRep(abs_sent_indices[i], rel_sent_indices_0_to_10[i],
            sent_lens[i], sent_representations_separate[i], cluster_rep_sent_separate, dist_separate[i]))
    return sent_reps

def features_to_array(sent_reps, features_list):
    x = []
    for rep in sent_reps:
        x_i = []
        for feature in features_list:
            val = getattr(rep, feature)
            if util.is_list_type(val):
                x_i.extend(val)
            else:
                x_i.append(val)
        x.append(x_i)
    return np.array(x)

def get_relative_sent_indices(sent_indices):
    relative_sent_indices = []
    prev_idx = -1
    cur_sent_indices = []
    for idx in sent_indices:
        if idx <= prev_idx:
            relative = [float(i)/len(cur_sent_indices) for i in cur_sent_indices]
            relative_sent_indices += relative
            prev_idx = -1
            cur_sent_indices = []
        cur_sent_indices.append(idx)
        prev_idx = idx
    if len(cur_sent_indices) > 0:
        relative = [float(i)/len(cur_sent_indices) for i in cur_sent_indices]
        relative_sent_indices += relative
    # if not FLAGS.normalize_features:
    relative_sent_indices_0_to_10 = [int(idx * 10) for idx in relative_sent_indices]
    return relative_sent_indices_0_to_10


def get_sent_indices(enc_sentences, doc_indices):
    cur_doc_idx = 0
    cur_sent_idx = 1
    count = 0
    sent_indices = []
    for sent_idx, sent in enumerate(enc_sentences):
        if cur_doc_idx != doc_indices[count]:
            cur_doc_idx = doc_indices[count]
            cur_sent_idx = 1
        sent_indices.append(cur_sent_idx)
        for word_idx, word in enumerate(sent):
            count += 1
        cur_sent_idx += 1
    return sent_indices

def get_sent_lens(enc_sentences):
    sent_lens = [len(sent) for sent in enc_sentences]
    return sent_lens


def get_ROUGE_Ls(art_oovs, all_original_abstracts_sents, vocab, enc_tokens):
    human_tokens = get_tokens_for_human_summaries(art_oovs, all_original_abstracts_sents, vocab, split_sents=False)  # list (of 4 human summaries) of list of token ids
    metric = 'recall'
    importances_hat = util.rouge_l_similarity(enc_tokens, human_tokens, vocab, metric=metric)
    importances = util.special_squash(importances_hat)
    return importances, importances_hat

def get_best_ROUGE_L_for_each_abs_sent(art_oovs, all_original_abstracts_sents, vocab, enc_tokens):
    human_tokens = get_tokens_for_human_summaries(art_oovs, all_original_abstracts_sents, vocab, split_sents=True)  # list (of 4 human summaries) of list of token ids
    if len(human_tokens) > 1:
        raise Exception('human_tokens (len %d) should have 1 entry, because cnn/dm has one abstract per article.' % len(human_tokens))
    human_tokens = human_tokens[0]
    metric = 'recall'
    similarity_matrix = util.rouge_l_similarity_matrix(enc_tokens, human_tokens, vocab, metric=metric)
    best_indices = []
    for col_idx in range(similarity_matrix.shape[1]):
        col = similarity_matrix[:,col_idx]
        sorted_indices = np.argsort(col)[::-1]
        idx = 0
        while sorted_indices[idx] in best_indices:
            idx += 1
            if idx >= len(sorted_indices):       #  If all sentences have been used then just continue
                idx = 0
                break
        best_idx = sorted_indices[idx]
        best_indices.append(best_idx)
    binary_y = np.zeros([len(enc_tokens)], dtype=float)
    binary_y[best_indices] = 1
    return binary_y

def get_tokens_for_human_summaries(art_oovs, all_original_abstracts_sents, vocab, split_sents=False):
    def get_all_summ_tokens(all_summs):
        return [get_summ_tokens(summ) for summ in all_summs]
    def get_summ_tokens(summ):
        summ_tokens = [get_sent_tokens(sent) for sent in summ]
        if split_sents:
            return summ_tokens
        else:
            return list(itertools.chain.from_iterable(summ_tokens))     # combines all sentences into one list of tokens for summary
    def get_sent_tokens(sent):
        words = sent.split()
        return data.abstract2ids(words, vocab, art_oovs)
    human_summaries = all_original_abstracts_sents
    all_summ_tokens = get_all_summ_tokens(human_summaries)
    return all_summ_tokens

def get_cluster_representations(sent_representations_separate):
    cluster_rep_sent_separate = np.mean(sent_representations_separate, axis=0)
    dist_separate = np.squeeze(cosine_similarity(sent_representations_separate, [cluster_rep_sent_separate]))
    return cluster_rep_sent_separate, dist_separate


def tokens_to_continuous_text(tokens, vocab, art_oovs):
    words = data.outputids2words(tokens, vocab, art_oovs)
    text = ' '.join(words)
    # text = text.decode('utf8')
    split_text = text.split(' ')
    if len(split_text) != len(words):
        for i in range(min(len(words), len(split_text))):
            try:
                print('%s\t%s'%(words[i], split_text[i]))
            except:
                print('FAIL\tFAIL')
        raise Exception('text ('+str(len(text.split()))+
                        ') does not have the same number of tokens as words ('+str(len(words))+')')

    return text

def get_sentence_splits(enc_sentences):
    '''Returns a list of indices, representing the word index for the first word of each sentence'''
    cur_idx = 0
    indices = []
    for sent in enc_sentences:
        indices.append(cur_idx)
        cur_idx += len(sent)
    return indices

def get_fw_bw_rep(enc_states, start_idx, end_idx):
    fw_state_size = enc_states.shape[1] // 2
    assert fw_state_size * 2 == enc_states.shape[1]
    fw_sent_rep = enc_states[end_idx, :fw_state_size]
    bw_sent_rep = enc_states[start_idx, fw_state_size:]
    rep = np.concatenate([fw_sent_rep, bw_sent_rep])
    return rep

def get_separate_enc_states(model, sess, enc_sentences, vocab, hps):
    reps = []
    examples = []
    for enc_sent in enc_sentences:
        sent_str = ' '.join(enc_sent)
        doc_indices = [0] * len(enc_sent)                   # just filler, shouldn't do anything
        ex = batcher.Example(sent_str, [], [[]], doc_indices, None, vocab, hps)
        examples.append(ex)
    chunks = util.chunks(examples, hps.batch_size)
    if len(chunks[-1]) != hps.batch_size:                   # If last chunk is not filled, then just artificially fill it
        for i in range(hps.batch_size - len(chunks[-1])):
            chunks[-1].append(examples[-1])
    for chunk in chunks:
        batch = batcher.Batch(chunk, hps, vocab)
        batch_enc_states, _ = model.run_encoder(sess, batch)
        for batch_idx, enc_states in enumerate(batch_enc_states):
            start_idx = 0
            end_idx = batch.enc_lens[batch_idx] - 1
            rep = get_fw_bw_rep(enc_states, start_idx, end_idx)
            reps.append(rep)
    reps = reps[:len(enc_sentences)]                        # Removes the filler examples
    return reps

def run_training(x, y):
    print("Starting SVR training")
    if FLAGS.importance_fn == 'svr':
        clf = svm.SVR()

    clf.fit(x, y)
    return clf

def load_data(data_path, num_instances):
    print('Loading SVR data')
    filelist = glob.glob(data_path) # get the list of datafiles
    assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
    filelist = sorted(filelist)
    instances = []
    for file_name in tqdm(filelist):
        with open(file_name) as f:
            examples = pickle.load(f)
        if num_instances == -1:
            num_instances = np.inf
        remaining_number = num_instances - sum([len(b) for b in instances])
        if len(examples) < remaining_number:
            instances.extend(examples)
        else:
            instances.extend(examples[:remaining_number])
            break
    print('Finished loading data. Number of instances=%d' % len(instances))
    return instances