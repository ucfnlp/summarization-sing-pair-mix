# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
# Modifications made 2018 by Logan Lebanoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to run beam search decoding"""
import numpy as np
import data
from absl import flags
import pg_mmr_functions
import util

FLAGS = flags.FLAGS


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage, mmr, summ_sent_idx, already_added):
        """Hypothesis constructor.

        Args:
            tokens: List of integers. The ids of the tokens that form the summary so far.
            log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
            state: Current state of the decoder, a LSTMStateTuple.
            attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
            p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
            coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.p_gens = p_gens
        self.coverage = coverage
        self.similarity = 0.
        self.mmr = mmr
        self.summ_sent_idx = summ_sent_idx
        self.already_added = already_added

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage, mmr, summ_sent_idx):
        """Return a NEW hypothesis, extended with the information from the latest step of beam search.

        Args:
            token: Integer. Latest token produced by beam search.
            log_prob: Float. Log prob of the latest token.
            state: Current decoder state, a LSTMStateTuple.
            attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
            p_gen: Generation probability on latest step. Float.
            coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
        Returns:
            New Hypothesis for next step.
        """
        if self.does_trigram_exist(token):
            log_prob = -1000
        return Hypothesis(tokens=self.tokens + [token],
                          log_probs=self.log_probs + [log_prob],
                          state=state,
                          attn_dists=self.attn_dists + [attn_dist],
                          p_gens=self.p_gens + [p_gen],
                          coverage=coverage,
                          mmr=mmr,
                          summ_sent_idx=summ_sent_idx,
                          already_added=self.already_added)

    def does_trigram_exist(self, token):
        if len(self.tokens) < 2:
            return False
        candidate_trigram = self.tokens[-2:] + [token]
        for i in range(len(self.tokens)-2):
            if self.tokens[i:i+3] == candidate_trigram:
                return True
        return False

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.log_prob / len(self.tokens)

def results_still_too_small(results):
    if FLAGS.better_beam_search:
        return True
    else:
        return len(results) < FLAGS.beam_size

def run_beam_search(sess, model, vocab, batch, ex_index, hps):
    """Performs beam search decoding on the given example.

    Args:
        sess: a tf.Session
        model: a seq2seq model
        vocab: Vocabulary object
        batch: Batch object that is the same example repeated across the batch

    Returns:
        best_hyp: Hypothesis object; the best hypothesis found by beam search.
    """

    max_dec_steps = FLAGS.max_dec_steps
    # Run the encoder to get the encoder hidden states and decoder initial state
    enc_states, dec_in_state = model.run_encoder(sess, batch)
    # dec_in_state is a LSTMStateTuple
    # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].

    # Sentence importance
    enc_sentences, enc_tokens = batch.tokenized_sents[0], batch.word_ids_sents[0]
    if FLAGS.ssi_data_path != '':   # if we are running on pg_mmr and bert
        mmr_init = None
    else:
        importances = pg_mmr_functions.get_importances(model, batch, enc_states, vocab, sess, hps)
        mmr_init = importances


    # Initialize beam_size-many hyptheses
    hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                       log_probs=[0.0],
                       state=dec_in_state,
                       attn_dists=[],
                       p_gens=[],
                       coverage=np.zeros([batch.enc_batch.shape[1]]),  # zero vector of length attention_length
                       mmr=mmr_init,
                       summ_sent_idx=0,
                       already_added=False
                       ) for hyp_idx in range(FLAGS.beam_size)]
    results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)


    steps = 0
    while steps < max_dec_steps and results_still_too_small(results):

        latest_tokens = [h.latest_token for h in hyps]  # latest token produced by each hypothesis
        latest_tokens = [t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in
                         latest_tokens]  # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings

        states = [h.state for h in hyps]  # list of current decoder states of the hypotheses
        prev_coverage = [h.coverage for h in hyps]  # list of coverage vectors (or None)

        # Mute all source sentences except the top k sentences
        prev_mmr = [h.mmr for h in hyps]
        if FLAGS.pg_mmr:
            if FLAGS.ssi_data_path != '':       # if we are doing pg_mmr with bert
                prev_mmr_for_words_list = []
                for batch_idx in range(len(batch.ssi_masks_padded)):
                    summ_sent_idx = hyps[batch_idx].summ_sent_idx
                    prev_sent_idx = max(0, summ_sent_idx-1)
                    if summ_sent_idx >= len(batch.ssi_masks_padded[batch_idx]):
                        print ("Performing modulo on summ_sent_idx (%d) because it has generated too many sentences." % summ_sent_idx)
                        summ_sent_idx = summ_sent_idx % len(batch.ssi_masks_padded[batch_idx])
                        prev_sent_idx = prev_sent_idx % len(batch.ssi_masks_padded[batch_idx])
                    prev_mmr_for_words_list.append([batch.ssi_masks_padded[batch_idx][prev_sent_idx], batch.ssi_masks_padded[batch_idx][summ_sent_idx]])
                prev_mmr_for_words = np.array(prev_mmr_for_words_list)
            else:
                if FLAGS.mute_k != -1:
                    prev_mmr = [pg_mmr_functions.mute_all_except_top_k(mmr, FLAGS.mute_k) for mmr in prev_mmr]
                prev_mmr_for_words = [pg_mmr_functions.convert_to_word_level(mmr, enc_tokens) for mmr in prev_mmr]
        else:
            prev_mmr_for_words = [None for _ in prev_mmr]


        # Run one step of the decoder to get the new info
        (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage, pre_attn_dists) = model.decode_onestep(sess=sess,
                                                                                                        batch=batch,
                                                                                                        latest_tokens=latest_tokens,
                                                                                                        enc_states=enc_states,
                                                                                                        dec_init_states=states,
                                                                                                        prev_coverage=prev_coverage,
                                                                                                        mmr_score=prev_mmr_for_words)

        # Extend each hypothesis and collect them all in all_hyps
        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(
            hyps)  # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
        for i in range(num_orig_hyps):
            h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], \
                                                             new_coverage[
                                                                 i]  # take the ith hypothesis and new decoder state info
            for j in range(FLAGS.beam_size * 2):  # for each of the top 2*beam_size hyps:
                # Extend the ith hypothesis with the jth option
                new_hyp = h.extend(token=topk_ids[i, j],
                                   log_prob=topk_log_probs[i, j],
                                   state=new_state,
                                   attn_dist=attn_dist,
                                   p_gen=p_gen,
                                   coverage=new_coverage_i,
                                   mmr=h.mmr,
                                   summ_sent_idx=h.summ_sent_idx)
                all_hyps.append(new_hyp)

        # Filter and collect any hypotheses that have produced the end token.
        hyps = []  # will contain hypotheses for the next step
        for h in sort_hyps(all_hyps):  # in order of most likely h
            if h.latest_token == vocab.word2id(data.STOP_DECODING):  # if stop token is reached...
                # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                if steps >= FLAGS.min_dec_steps:
                    results.append(h)
                    h.already_added = True
                    # print 'ADDED THING'
            else:  # hasn't reached stop token, so continue to extend this hypothesis
                hyps.append(h)
            if len(hyps) == FLAGS.beam_size or (not FLAGS.better_beam_search and len(results) == FLAGS.beam_size):
                # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop. (Unless it's Logan's better beam search)
                break

        # Update the MMR scores when a sentence is completed
        if FLAGS.pg_mmr:
            for hyp_idx, hyp in enumerate(hyps):
                if hyp.latest_token == vocab.word2id(data.PERIOD):     # if in regular mode, and the hyp ends in a period
                    if FLAGS.ssi_data_path != '':       # if we are doing pg_mmr with bert
                        hyp.summ_sent_idx += 1
                        # If we have exhausted the singletons and pairs from BERT, then put in results
                        if hyp.summ_sent_idx >= len(batch.ssis[hyp_idx]) and not hyp.already_added:
                            results.append(hyp)
                            hyp.already_added = True
                            # print 'ADDED THING 2'
                    else:
                        pg_mmr_functions.update_similarity_and_mmr(hyp, importances, batch, enc_tokens, vocab)

        steps += 1

    # At this point, either we've got beam_size results, or we've reached maximum decoder steps

    if len(results) == 0:  # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
        results = hyps

    # Sort hypotheses by average log probability
    hyps_sorted = sort_hyps(results)
    best_hyp = hyps_sorted[0]

    # Save plots of the distributions (importance, similarity, mmr)
    if FLAGS.plot_distributions and FLAGS.pg_mmr:
        pg_mmr_functions.save_distribution_plots(importances, enc_sentences,
                                   enc_tokens, best_hyp, batch, vocab, ex_index)


    # Return the hypothesis with highest average log prob
    return best_hyp


def sort_hyps(hyps):
    """Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
