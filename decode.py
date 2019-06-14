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

"""This file contains code to run beam search decoding, including running ROUGE evaluation and producing JSON datafiles for the in-browser attention visualizer, which can be found here https://github.com/abisee/attn_vis"""
import glob
import os
import time

import pickle
import tensorflow as tf
import beam_search
import data
import json
import pyrouge
import util
from sumy.nlp.tokenizers import Tokenizer
from tqdm import tqdm
from absl import flags
from absl import logging
import logging as log
import rouge_functions

import importance_features
import convert_data
from batcher import create_batch
import attn_selections
import ssi_functions

FLAGS = flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60	# max number of seconds before loading new checkpoint
threshold = 0.5
prob_to_keep = 0.33


class BeamSearchDecoder(object):
    """Beam search decoder."""

    def __init__(self, model, batcher, vocab):
        """Initialize decoder.

        Args:
            model: a Seq2SeqAttentionModel object.
            batcher: a Batcher object.
            vocab: Vocabulary object
        """
        self._model = model
        self._model.build_graph()
        self._batcher = batcher
        self._vocab = vocab
        self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
        self._sess = tf.Session(config=util.get_config())

        # Load an initial checkpoint to use for decoding
        ckpt_path = util.load_ckpt(self._saver, self._sess)

        if FLAGS.single_pass:
            # Make a descriptive decode directory name
            ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
            self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
# 			if os.path.exists(self._decode_dir):
# 				raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

        else: # Generic decode dir name
            self._decode_dir = os.path.join(FLAGS.log_root, "decode")

        # Make the decode dir if necessary
        if not os.path.exists(self._decode_dir): os.makedirs(self._decode_dir)

        if FLAGS.single_pass:
            # Make the dirs to contain output written in the correct format for pyrouge
            self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
            if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
            self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
            if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)
            self._human_dir = os.path.join(self._decode_dir, "human_readable")
            if not os.path.exists(self._human_dir): os.mkdir(self._human_dir)
            self._highlight_dir = os.path.join(self._decode_dir, "highlight")
            if not os.path.exists(self._highlight_dir): os.mkdir(self._highlight_dir)


    def decode(self):
        """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
        t0 = time.time()
        counter = 0
        attn_dir = os.path.join(self._decode_dir, 'attn_vis_data')
        total = len(glob.glob(self._batcher._data_path)) * 1000
        pbar = tqdm(total=total)
        while True:
            batch = self._batcher.next_batch()	# 1 example repeated across batch
            if batch is None: # finished decoding dataset in single_pass mode
                assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
                attn_selections.process_attn_selections(attn_dir, self._decode_dir, self._vocab)
                logging.info("Decoder has finished reading dataset for single_pass.")
                logging.info("Output has been saved in %s and %s.", self._rouge_ref_dir, self._rouge_dec_dir)
                if len(os.listdir(self._rouge_ref_dir)) != 0:
                    logging.info("Now starting ROUGE eval...")
                    results_dict = rouge_functions.rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
                    rouge_functions.rouge_log(results_dict, self._decode_dir)
                return

            original_article = batch.original_articles[0]	# string
            original_abstract = batch.original_abstracts[0]	# string
            all_original_abstract_sents = batch.all_original_abstracts_sents[0]
            raw_article_sents = batch.raw_article_sents[0]

            article_withunks = data.show_art_oovs(original_article, self._vocab) # string
            abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

            decoded_words, decoded_output, best_hyp = decode_example(self._sess, self._model, self._vocab, batch, counter, self._batcher._hps)

            if FLAGS.single_pass:
                if counter < 1000:
                    self.write_for_human(raw_article_sents, all_original_abstract_sents, decoded_words, counter)
                rouge_functions.write_for_rouge(all_original_abstract_sents, None, counter, self._rouge_ref_dir, self._rouge_dec_dir, decoded_words=decoded_words) # write ref summary and decoded summary to file, to eval with pyrouge later
                if FLAGS.attn_vis:
                    self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens, counter) # write info to .json file for visualization tool

                    if counter % 1000 == 0:
                        attn_selections.process_attn_selections(attn_dir, self._decode_dir, self._vocab)

                counter += 1 # this is how many examples we've decoded
            else:
                print_results(article_withunks, abstract_withunks, decoded_output) # log output to screen
                self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens, counter) # write info to .json file for visualization tool

                # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
                t1 = time.time()
                if t1-t0 > SECS_UNTIL_NEW_CKPT:
                    logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint', t1-t0)
                    _ = util.load_ckpt(self._saver, self._sess)
                    t0 = time.time()
            pbar.update(1)
        pbar.close()

    def decode_iteratively(self, example_generator, total, names_to_types, ssi_list, hps):
        attn_vis_idx = 0
        for example_idx, example in enumerate(tqdm(example_generator, total=total)):
            raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, groundtruth_article_lcs_paths_list = util.unpack_tf_example(
                example, names_to_types)
            article_sent_tokens = [util.process_sent(sent) for sent in raw_article_sents]
            groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]

            if ssi_list is None:    # this is if we are doing the upper bound evaluation (ssi_list comes straight from the groundtruth)
                sys_ssi = groundtruth_similar_source_indices_list
                if FLAGS.singles_and_pairs == 'singles':
                    sys_ssi = util.enforce_sentence_limit(sys_ssi, 1)
                elif FLAGS.singles_and_pairs == 'both':
                    sys_ssi = util.enforce_sentence_limit(sys_ssi, 2)
                sys_ssi = util.replace_empty_ssis(sys_ssi, raw_article_sents)
            else:
                gt_ssi, sys_ssi, ext_len = ssi_list[example_idx]
                if FLAGS.singles_and_pairs == 'singles':
                    sys_ssi = util.enforce_sentence_limit(sys_ssi, 1)
                    groundtruth_similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, 1)
                    gt_ssi = util.enforce_sentence_limit(gt_ssi, 1)
                elif FLAGS.singles_and_pairs == 'both':
                    sys_ssi = util.enforce_sentence_limit(sys_ssi, 2)
                    groundtruth_similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, 2)
                    gt_ssi = util.enforce_sentence_limit(gt_ssi, 2)
                if gt_ssi != groundtruth_similar_source_indices_list:
                    raise Exception('Example %d has different groundtruth source indices: ' + str(groundtruth_similar_source_indices_list) + ' || ' + str(gt_ssi))
                if FLAGS.dataset_name == 'xsum':
                    sys_ssi = [sys_ssi[0]]

            final_decoded_words = []
            final_decoded_outpus = ''
            best_hyps = []
            highlight_html_total = ''
            for ssi_idx, ssi in enumerate(sys_ssi):
                selected_article_lcs_paths = None
                # selected_article_lcs_paths = article_lcs_paths_list[ssi_idx]
                # ssi, selected_article_lcs_paths = util.make_ssi_chronological(ssi, selected_article_lcs_paths)
                # selected_article_lcs_paths = [selected_article_lcs_paths]
                selected_raw_article_sents = util.reorder(raw_article_sents, ssi)
                selected_article_text = ' '.join( [' '.join(sent) for sent in util.reorder(article_sent_tokens, ssi)] )
                selected_doc_indices_str = '0 ' * len(selected_article_text.split())
                if FLAGS.upper_bound:
                    selected_groundtruth_summ_sent = [[groundtruth_summ_sents[0][ssi_idx]]]
                else:
                    selected_groundtruth_summ_sent = groundtruth_summ_sents

                batch = create_batch(selected_article_text, selected_groundtruth_summ_sent, selected_doc_indices_str, selected_raw_article_sents, selected_article_lcs_paths, FLAGS.batch_size, hps, self._vocab)

                original_article = batch.original_articles[0]  # string
                original_abstract = batch.original_abstracts[0]  # string
                article_withunks = data.show_art_oovs(original_article, self._vocab)  # string
                abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab,
                                                       (batch.art_oovs[0] if FLAGS.pointer_gen else None))  # string
                # article_withunks = data.show_art_oovs(original_article, self._vocab) # string
                # abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

                if FLAGS.first_intact and ssi_idx == 0:
                    decoded_words = selected_article_text.strip().split()
                    decoded_output = selected_article_text
                else:
                    decoded_words, decoded_output, best_hyp = decode_example(self._sess, self._model, self._vocab, batch, example_idx, hps)
                    best_hyps.append(best_hyp)
                final_decoded_words.extend(decoded_words)
                final_decoded_outpus += decoded_output

                if example_idx < 1000:
                    min_matched_tokens = 2
                    selected_article_sent_tokens = [util.process_sent(sent) for sent in selected_raw_article_sents]
                    highlight_summary_sent_tokens = [decoded_words]
                    highlight_ssi_list, lcs_paths_list, highlight_article_lcs_paths_list, highlight_smooth_article_lcs_paths_list = ssi_functions.get_simple_source_indices_list(
                        highlight_summary_sent_tokens,
                        selected_article_sent_tokens, None, 2, min_matched_tokens)
                    highlighted_html = ssi_functions.html_highlight_sents_in_article(highlight_summary_sent_tokens,
                                                                                   highlight_ssi_list,
                                                                                     selected_article_sent_tokens,
                                                                                   lcs_paths_list=lcs_paths_list,
                                                                                   article_lcs_paths_list=highlight_smooth_article_lcs_paths_list)
                    highlight_html_total += '<u>System Summary</u><br><br>' + highlighted_html + '<br><br>'

                if FLAGS.attn_vis and example_idx < 200:
                    self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists,
                                           best_hyp.p_gens,
                                           attn_vis_idx)  # write info to .json file for visualization tool
                    attn_vis_idx += 1

                if len(final_decoded_words) >= 100:
                    break

            if example_idx < 1000:
                self.write_for_human(raw_article_sents, groundtruth_summ_sents, final_decoded_words, example_idx)
                ssi_functions.write_highlighted_html(highlight_html_total, self._highlight_dir, example_idx)

            # if example_idx % 100 == 0:
            #     attn_dir = os.path.join(self._decode_dir, 'attn_vis_data')
            #     attn_selections.process_attn_selections(attn_dir, self._decode_dir, self._vocab)

            rouge_functions.write_for_rouge(groundtruth_summ_sents, None, example_idx, self._rouge_ref_dir, self._rouge_dec_dir, decoded_words=final_decoded_words, log=False) # write ref summary and decoded summary to file, to eval with pyrouge later
            # if FLAGS.attn_vis:
            #     self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens, example_idx) # write info to .json file for visualization tool
            example_idx += 1 # this is how many examples we've decoded

        logging.info("Decoder has finished reading dataset for single_pass.")
        logging.info("Output has been saved in %s and %s.", self._rouge_ref_dir, self._rouge_dec_dir)
        if len(os.listdir(self._rouge_ref_dir)) != 0:
            if FLAGS.dataset_name == 'xsum':
                l_param = 100
            else:
                l_param = 100
            logging.info("Now starting ROUGE eval...")
            results_dict = rouge_functions.rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir, l_param=l_param)
            rouge_functions.rouge_log(results_dict, self._decode_dir)



    def write_for_human(self, raw_article_sents, all_reference_sents, decoded_words, ex_index):

        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index(".")
            except ValueError: # there is text remaining that doesn't end in "."
                fst_period_idx = len(decoded_words)
            sent = decoded_words[:fst_period_idx+1] # sentence up to and including the period
            decoded_words = decoded_words[fst_period_idx+1:] # everything else
            decoded_sents.append(' '.join(sent))

        # pyrouge calls a perl script that puts the data into HTML files.
        # Therefore we need to make our output HTML safe.
        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        all_reference_sents = [[make_html_safe(w) for w in abstract] for abstract in all_reference_sents]

        # Write to file
        human_file = os.path.join(self._human_dir, '%06d_human.txt' % ex_index)

        with open(human_file, "w") as f:
            f.write('Human Summary:\n--------------------------------------------------------------\n')
            for abs_idx, abs in enumerate(all_reference_sents):
                for idx,sent in enumerate(abs):
                    f.write(sent+"\n")
            f.write('\nSystem Summary:\n--------------------------------------------------------------\n')
            for sent in decoded_sents:
                f.write(sent + '\n')
            f.write('\nArticle:\n--------------------------------------------------------------\n')
            for sent in raw_article_sents:
                f.write(sent + '\n')

    def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens, ex_index, ssi=None):
        """Write some data to json file, which can be read into the in-browser attention visualizer tool:
            https://github.com/abisee/attn_vis

        Args:
            article: The original article string.
            abstract: The human (correct) abstract string.
            attn_dists: List of arrays; the attention distributions.
            decoded_words: List of strings; the words of the generated summary.
            p_gens: List of scalars; the p_gen values. If not running in pointer-generator mode, list of None.
        """
        article_lst = article.split() # list of words
        decoded_lst = decoded_words # list of decoded words
        to_write = {
                'article_lst': [make_html_safe(t) for t in article_lst],
                'decoded_lst': [make_html_safe(t) for t in decoded_lst],
                'abstract_str': make_html_safe(abstract),
                'attn_dists': attn_dists
        }
        if FLAGS.pointer_gen:
            to_write['p_gens'] = p_gens
        if ssi is not None:
            to_write['ssi'] = ssi
        util.create_dirs(os.path.join(self._decode_dir, 'attn_vis_data'))
        output_fname = os.path.join(self._decode_dir, 'attn_vis_data', '%06d.json' % ex_index)
        with open(output_fname, 'w') as output_file:
            json.dump(to_write, output_file)
        # logging.info('Wrote visualization data to %s', output_fname)

    def calc_importance_features(self, data_path, hps, model_save_path, docs_desired):
        """Calculate sentence-level features and save as a dataset"""
        data_path_filter_name = os.path.basename(data_path)
        if 'train' in data_path_filter_name:
            data_split = 'train'
        elif 'val' in data_path_filter_name:
            data_split = 'val'
        elif 'test' in data_path_filter_name:
            data_split = 'test'
        else:
            data_split = 'feats'
        if 'cnn-dailymail' in data_path:
            inst_per_file = 1000
        else:
            inst_per_file = 1
        filelist = glob.glob(data_path)
        num_documents_desired = docs_desired
        pbar = tqdm(initial=0, total=num_documents_desired)

        instances = []
        sentences = []
        counter = 0
        doc_counter = 0
        file_counter = 0
        while True:
            batch = self._batcher.next_batch()	# 1 example repeated across batch
            if doc_counter >= num_documents_desired:
                save_path = os.path.join(model_save_path, data_split + '_%06d'%file_counter)
                with open(save_path, 'wb') as f:
                    pickle.dump(instances, f)
                print(('Saved features at %s' % save_path))
                return

            if batch is None: # finished decoding dataset in single_pass mode
                raise Exception('We havent reached the num docs desired (%d), instead we reached (%d)' % (num_documents_desired, doc_counter))


            batch_enc_states, _ = self._model.run_encoder(self._sess, batch)
            for batch_idx, enc_states in enumerate(batch_enc_states):
                art_oovs = batch.art_oovs[batch_idx]
                all_original_abstracts_sents = batch.all_original_abstracts_sents[batch_idx]

                tokenizer = Tokenizer('english')
                # List of lists of words
                enc_sentences, enc_tokens = batch.tokenized_sents[batch_idx], batch.word_ids_sents[batch_idx]
                enc_sent_indices = importance_features.get_sent_indices(enc_sentences, batch.doc_indices[batch_idx])
                enc_sentences_str = [' '.join(sent) for sent in enc_sentences]

                sent_representations_separate = importance_features.get_separate_enc_states(self._model, self._sess, enc_sentences, self._vocab, hps)

                sent_indices = enc_sent_indices
                sent_reps = importance_features.get_importance_features_for_article(
                    enc_states, enc_sentences, sent_indices, tokenizer, sent_representations_separate)
                y, y_hat = importance_features.get_ROUGE_Ls(art_oovs, all_original_abstracts_sents, self._vocab, enc_tokens)
                binary_y = importance_features.get_best_ROUGE_L_for_each_abs_sent(art_oovs, all_original_abstracts_sents, self._vocab, enc_tokens)
                for rep_idx, rep in enumerate(sent_reps):
                    rep.y = y[rep_idx]
                    rep.binary_y = binary_y[rep_idx]

                for rep_idx, rep in enumerate(sent_reps):
                    # Keep all sentences with importance above threshold. All others will be kept with a probability of prob_to_keep
                    if FLAGS.importance_fn == 'svr':
                        instances.append(rep)
                        sentences.append(sentences)
                        counter += 1 # this is how many examples we've decoded
            doc_counter += len(batch_enc_states)
            pbar.update(len(batch_enc_states))



def decode_example(sess, model, vocab, batch, counter, hps):
    # Run beam search to get best Hypothesis
    best_hyp = beam_search.run_beam_search(sess, model, vocab, batch, counter, hps)

    # Extract the output ids from the hypothesis and convert back to words
    output_ids = [int(t) for t in best_hyp.tokens[1:]]
    decoded_words = data.outputids2words(output_ids, vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))

    # Remove the [STOP] token from decoded_words, if necessary
    try:
        fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
        decoded_words = decoded_words[:fst_stop_idx]
    except ValueError:
        decoded_words = decoded_words
    decoded_output = ' '.join(decoded_words) # single string
    return decoded_words, decoded_output, best_hyp


def print_results(article, abstract, decoded_output):
    """Prints the article, the reference summmary and the decoded summary to screen"""
    print("")
    logging.info('ARTICLE:	%s', article)
    logging.info('REFERENCE SUMMARY: %s', abstract)
    logging.info('GENERATED SUMMARY: %s', decoded_output)
    print("")


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def get_decode_dir_name(ckpt_name):
    """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

    if "train" in FLAGS.data_path: dataset = "train"
    elif "val" in FLAGS.data_path: dataset = "val"
    elif "test" in FLAGS.data_path: dataset = "test"
    else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
    dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
    if ckpt_name is not None:
        dirname += "_%s" % ckpt_name
    return dirname
