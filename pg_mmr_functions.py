import textwrap as tw
import PIL
import itertools
import util
from util import get_similarity, rouge_l_similarity
import importance_features
import dill
import time
import random
import numpy as np
import os
import data
from absl import flags
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import matplotlib
if not "DISPLAY" in os.environ:
    matplotlib.use("Agg")
from matplotlib import pyplot as plt

FLAGS = flags.FLAGS

def get_summ_sents_and_tokens(summ_tokens, batch, vocab):
    summ_str = importance_features.tokens_to_continuous_text(summ_tokens, vocab, batch.art_oovs[0])
    sentences = util.tokenizer.to_sentences(summ_str)
    if data.PERIOD not in sentences[-1]:
        sentences = sentences[:len(sentences) - 1]  # Doesn't include the last sentence if incomplete (no period)
    sent_words = []
    sent_tokens = []
    token_idx = 0
    for sentence in sentences:
        words = sentence.split(' ')
        sent_words.append(words)
        tokens = summ_tokens[token_idx:token_idx + len(words)]
        sent_tokens.append(tokens)
        token_idx += len(words)
    return sent_words, sent_tokens

def convert_to_word_level(mmr_for_sentences, enc_tokens):
    num_tokens = len(util.flatten_list_of_lists(enc_tokens))
    mmr = np.ones([num_tokens], dtype=float) / num_tokens
    # Calculate how much for each word in source
    word_idx = 0
    for sent_idx in range(len(enc_tokens)):
        mmr_for_words = np.full([len(enc_tokens[sent_idx])], mmr_for_sentences[sent_idx])
        mmr[word_idx:word_idx + len(mmr_for_words)] = mmr_for_words
        word_idx += len(mmr_for_words)
    return mmr

def calc_mmr_from_sim_and_imp(similarity, importances):
    new_mmr =  FLAGS.lambda_val*importances - (1-FLAGS.lambda_val)*similarity
    new_mmr = np.maximum(new_mmr, 0)
    return new_mmr

def mute_all_except_top_k(array, k):
    num_reservoirs_still_full = np.sum(array > 0)
    if num_reservoirs_still_full < k:
        selected_indices = np.nonzero(array)
    else:
        selected_indices = array.argsort()[::-1][:k]
    res = np.zeros_like(array, dtype=float)
    for selected_idx in selected_indices:
        if FLAGS.retain_mmr_values:
            res[selected_idx] = array[selected_idx]
        else:
            res[selected_idx] = 1.
    return res

def get_tokens_for_human_summaries(batch, vocab):
    art_oovs = batch.art_oovs[0]
    def get_all_summ_tokens(all_summs):
        return [get_summ_tokens(summ) for summ in all_summs]
    def get_summ_tokens(summ):
        summ_tokens = [get_sent_tokens(sent) for sent in summ]
        return list(itertools.chain.from_iterable(summ_tokens))     # combines all sentences into one list of tokens for summary
    def get_sent_tokens(sent):
        words = sent.split()
        return data.abstract2ids(words, vocab, art_oovs)
    human_summaries = batch.all_original_abstracts_sents[0]
    all_summ_tokens = get_all_summ_tokens(human_summaries)
    return all_summ_tokens

def get_tfidf_importances(raw_article_sents):
    tfidf_model_path = os.path.join(FLAGS.actual_log_root, 'tfidf_vectorizer', FLAGS.original_dataset_name + '.dill')

    while True:
        try:
            with open(tfidf_model_path, 'rb') as f:
                tfidf_vectorizer = dill.load(f)
            break
        except (EOFError, KeyError):
            time.sleep(random.randint(3,6))
            continue
    sent_reps = tfidf_vectorizer.transform(raw_article_sents)
    cluster_rep = np.mean(sent_reps, axis=0)
    similarity_matrix = cosine_similarity(sent_reps, cluster_rep)
    return np.squeeze(similarity_matrix)

def get_importances(model, batch, enc_states, vocab, sess, hps):
    if FLAGS.pg_mmr:
        enc_sentences, enc_tokens = batch.tokenized_sents[0], batch.word_ids_sents[0]
        if FLAGS.importance_fn == 'oracle':
            human_tokens = get_tokens_for_human_summaries(batch, vocab)     # list (of 4 human summaries) of list of token ids
            metric = 'recall'
            importances_hat = rouge_l_similarity(enc_tokens, human_tokens, vocab, metric=metric)
        elif FLAGS.importance_fn == 'tfidf':
            importances_hat = get_tfidf_importances(batch.raw_article_sents[0])
        importances = util.special_squash(importances_hat)
    else:
        importances = None
    return importances

def update_similarity_and_mmr(hyp, importances, batch, enc_tokens, vocab):
    summ_sents, summ_tokens = get_summ_sents_and_tokens(hyp.tokens, batch, vocab)
    hyp.similarity = get_similarity(enc_tokens, summ_tokens, vocab)
    hyp.mmr = calc_mmr_from_sim_and_imp(hyp.similarity, importances)