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

"""This file contains code to process data into batches"""
import os
import pickle

import queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import data
import nltk
import util
from absl import logging
import pg_mmr_functions

max_dec_sents = 10
chronological_ssi = True

class Example(object):
    """Class representing a train/val/test example for text summarization."""

    def __init__(self, article, abstract_sentences, all_abstract_sentences, doc_indices, raw_article_sents, ssi, vocab, hps):
        """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        Args:
            article: source text; a string. each token is separated by a single space.
            abstract_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
            vocab: Vocabulary object
            hps: hyperparameters
        """
        self.hps = hps

        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)


        # # Process the article
        # article_words = article.split()
        # if len(article_words) > hps.max_enc_steps:
        #     article_words = article_words[:hps.max_enc_steps]
        # self.enc_input = [vocab.word2id(w) for w in article_words] # list of word ids; OOVs are represented by the id for UNK token

        # Process the abstract
        abstract = ' '.join(abstract_sentences) # string
        abstract_words = abstract.split() # list of strings
        abs_ids = [vocab.word2id(w) for w in abstract_words] # list of word ids; OOVs are represented by the id for UNK token

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, hps.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if hps.pointer_gen:

            if raw_article_sents is not None and len(raw_article_sents) > 0:
                # self.tokenized_sents = [util.process_sent(sent) for sent in raw_article_sents]
                self.tokenized_sents = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]

                # Process the article
                article_words = util.flatten_list_of_lists(self.tokenized_sents)
                if len(article_words) > hps.max_enc_steps:
                    article_words = article_words[:hps.max_enc_steps]
                self.enc_input = [vocab.word2id(w) for w in
                                  article_words]  # list of word ids; OOVs are represented by the id for UNK token

                if len(all_abstract_sentences) == 1:
                    doc_indices = [0] * len(article_words)

                self.word_ids_sents, self.article_oovs = data.tokenizedarticle2ids(self.tokenized_sents, vocab)
                self.enc_input_extend_vocab = util.flatten_list_of_lists(self.word_ids_sents)
                if len(self.enc_input_extend_vocab) > hps.max_enc_steps:
                    self.enc_input_extend_vocab = self.enc_input_extend_vocab[:hps.max_enc_steps]
                self.enc_len = len(self.enc_input_extend_vocab) # store the length after truncation but before padding
            else:
                # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
                article_str = util.to_unicode(article)
                raw_article_sents = nltk.tokenize.sent_tokenize(article_str)
                self.tokenized_sents = [util.process_sent(sent) for sent in raw_article_sents]

                # Process the article
                article_words = util.flatten_list_of_lists(self.tokenized_sents)
                if len(article_words) > hps.max_enc_steps:
                    article_words = article_words[:hps.max_enc_steps]
                self.enc_input = [vocab.word2id(w) for w in
                                  article_words]  # list of word ids; OOVs are represented by the id for UNK token

                if len(all_abstract_sentences) == 1:
                    doc_indices = [0] * len(article_words)

                self.word_ids_sents, self.article_oovs = data.tokenizedarticle2ids(self.tokenized_sents, vocab)
                self.enc_input_extend_vocab = util.flatten_list_of_lists(self.word_ids_sents)
                # self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)
                if len(self.enc_input_extend_vocab) > hps.max_enc_steps:
                    self.enc_input_extend_vocab = self.enc_input_extend_vocab[:hps.max_enc_steps]
                self.enc_len = len(self.enc_input_extend_vocab) # store the length after truncation but before padding

            if self.hps.word_imp_reg:
                self.enc_importances = self.get_enc_importances(self.tokenized_sents, abstract_words)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, hps.max_dec_steps, start_decoding, stop_decoding)

        if ssi is not None:
            # Translate the similar source indices into masks over the encoder input
            self.ssi_masks = []
            for source_indices in ssi:
                ssi_sent_mask = [0.] * len(raw_article_sents)
                for source_idx in source_indices:
                    if source_idx >= len(ssi_sent_mask):
                        a=0
                    ssi_sent_mask[source_idx] = 1.
                ssi_mask = pg_mmr_functions.convert_to_word_level(ssi_sent_mask, self.tokenized_sents)
                self.ssi_masks.append(ssi_mask)

            summary_sent_tokens = [sent.strip().split() for sent in abstract_sentences]
            if self.hps.ssi_data_path is None and len(self.ssi_masks) != len(summary_sent_tokens):
                raise Exception('len(self.ssi_masks) != len(summary_sent_tokens)')

            self.sent_indices = pg_mmr_functions.convert_to_word_level(list(range(len(summary_sent_tokens))), summary_sent_tokens).tolist()


        # Store the original strings
        self.original_article = article
        self.raw_article_sents = raw_article_sents
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences
        self.all_original_abstract_sents = all_abstract_sentences

        self.doc_indices = doc_indices
        self.ssi = ssi

    def get_enc_importances(self, tokenized_sents, abstract_words):
        lemmatize = True
        if lemmatize:
            article_sent_tokens_lemma = util.lemmatize_sent_tokens(tokenized_sents)
            summary_sent_tokens_lemma = util.lemmatize_sent_tokens([abstract_words])
        article_tokens = util.flatten_list_of_lists(article_sent_tokens_lemma)
        abstract_tokens = util.flatten_list_of_lists(summary_sent_tokens_lemma)
        enc_importances = [1. if token in abstract_tokens else 0. for token in article_tokens]
        return enc_importances


    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).

        Args:
            sequence: List of ids (integers)
            max_len: integer
            start_id: integer
            stop_id: integer

        Returns:
            inp: sequence length <=max_len starting with start_id
            target: sequence same length as input, ending with stop_id only if there was no truncation
        """
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len: # truncate
            inp = inp[:max_len]
            target = target[:max_len] # no end_token
        else: # no truncation
            target.append(stop_id) # end token
        assert len(inp) == len(target)
        return inp, target


    def pad_decoder_inp_targ(self, max_len, pad_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)


    def pad_encoder_input(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        self.enc_input = self.enc_input[:max_len]       # TODO: This is a very hacky fix. This could be causing problems of misalignment between enc_input and enc_input_extend_vocab. Actually, the problem probably has to do with doing word tokenization on raw_article_sents (which turns into enc_input_ but then using article for enc_input
        if self.hps.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)
            self.enc_input_extend_vocab = self.enc_input_extend_vocab[:max_len]
        if self.hps.word_imp_reg:
            while len(self.enc_importances) < max_len:
                self.enc_importances.append(0.)
            self.enc_importances = self.enc_importances[:max_len]
        if self.hps.tag_tokens:
            while len(self.importance_mask) < max_len:
                self.importance_mask.append(0)
            self.importance_mask = self.importance_mask[:max_len]


    def pad_doc_indices(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.doc_indices) < max_len:
            self.doc_indices.append(pad_id)

    def pad_ssi_masks(self, max_len, max_enc_len, pad_id):
        self.ssi_masks_padded = []
        """Pad the encoder input sequence with pad_id up to max_len."""
        for idx in range(len(self.ssi_masks)):
            """Pad the encoder input sequence with pad_id up to max_len."""
            # self.ssi_masks[idx] = np.pad(self.ssi_masks[idx], (0, max_enc_len-len(self.ssi_masks[idx])), 'constant', constant_values=(pad_id, pad_id))
            if len(self.ssi_masks[idx]) < max_enc_len:
                to_add = np.pad(self.ssi_masks[idx], (0, max_enc_len-len(self.ssi_masks[idx])), 'constant', constant_values=(pad_id, pad_id))
            else:
                to_add = self.ssi_masks[idx][:max_enc_len]
            self.ssi_masks_padded.append(to_add)

        while len(self.ssi_masks_padded) < max_len:
            self.ssi_masks_padded.append([pad_id] * max_enc_len)
        self.ssi_masks_padded = self.ssi_masks_padded[:max_len]
        self.ssi_masks_padded = np.array(self.ssi_masks_padded)

    def pad_sent_indices(self, max_len, pad_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""
        while len(self.sent_indices) < max_len:
            self.sent_indices.append(pad_id)
        self.sent_indices = self.sent_indices[:max_len]

    def are_ssi_in_max_enc_steps(self, max_len):
        cur_token_idx = 0
        sent_idx_at_400 = 100000
        for sent_idx, sent_tokens in enumerate(self.tokenized_sents):
            for token in sent_tokens:
                cur_token_idx += 1
                if cur_token_idx >= max_len:
                    sent_idx_at_400 = sent_idx
                    break
            if cur_token_idx >= max_len:
                break
        for source_indices in self.ssi:
            for source_idx in source_indices:
                if source_idx >= sent_idx_at_400:
                    return False
        return True

    def fix_outside_and_empty_masks(self, max_enc_steps, max_enc_seq_len):
        cur_token_idx = 0
        sent_idx_at_400 = 100000
        for sent_idx, sent_tokens in enumerate(self.tokenized_sents):
            for token in sent_tokens:
                cur_token_idx += 1
                if cur_token_idx >= max_enc_steps:
                    sent_idx_at_400 = sent_idx
                    break
            if cur_token_idx >= max_enc_steps:
                break
        new_masks = []
        for source_indices_idx, source_indices in enumerate(self.ssi):
            should_make_all_ones = False
            for source_idx in source_indices:
                if source_idx >= sent_idx_at_400:
                    should_make_all_ones = True
            if len(source_indices) == 0:
                should_make_all_ones = True

            if should_make_all_ones:
                new_masks.append([1] * max_enc_steps)
            else:
                new_masks.append(self.ssi_masks[source_indices_idx])
        self.ssi_masks = new_masks



class Batch(object):
    """Class representing a minibatch of train/val/test examples for text summarization."""

    def __init__(self, example_list, hps, vocab):
        """Turns the example_list into a Batch object.

        Args:
             example_list: List of Example objects
             hps: hyperparameters
             vocab: Vocabulary object
        """
        self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list, hps) # initialize the input to the encoder
        self.init_decoder_seq(example_list, hps) # initialize the input and targets for the decoder
        if example_list[0].ssi is not None:
            self.init_ssi_masks(example_list, hps)
        self.store_orig_strings(example_list) # store the original strings

    def init_encoder_seq(self, example_list, hps):
        """Initializes the following:
                self.enc_batch:
                    numpy array of shape (batch_size, <=max_enc_steps) containing integer ids (all OOVs represented by UNK id), padded to length of longest sequence in the batch
                self.enc_lens:
                    numpy array of shape (batch_size) containing integers. The (truncated) length of each encoder input sequence (pre-padding).
                self.enc_padding_mask:
                    numpy array of shape (batch_size, <=max_enc_steps), containing 1s and 0s. 1s correspond to real tokens in enc_batch and target_batch; 0s correspond to padding.

            If hps.pointer_gen, additionally initializes the following:
                self.max_art_oovs:
                    maximum number of in-article OOVs in the batch
                self.art_oovs:
                    list of list of in-article OOVs (strings), for each example in the batch
                self.enc_batch_extend_vocab:
                    Same as self.enc_batch, but in-article OOVs are represented by their temporary article OOV number.
        """
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
            ex.pad_doc_indices(max_enc_seq_len, 0)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)
        self.doc_indices = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1
            self.doc_indices[i, :] = ex.doc_indices[:max_enc_seq_len]

        # For pointer-generator mode, need to store some extra info
        if hps.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

        if hps.word_imp_reg:
            self.enc_importances = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)
            for i, ex in enumerate(example_list):
                self.enc_importances[i, :] = ex.enc_importances[:]

        if hps.tag_tokens:
            self.importance_masks = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.importance_masks[i, :] = ex.importance_mask[:]


    def init_decoder_seq(self, example_list, hps):
        """Initializes the following:
                self.dec_batch:
                    numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.
                self.target_batch:
                    numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.
                self.dec_padding_mask:
                    numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
                """
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
        self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def init_ssi_masks(self, example_list, hps):
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        for ex in example_list:
            ex.fix_outside_and_empty_masks(hps.max_enc_steps, max_enc_seq_len)
        for ex_idx, ex in enumerate(example_list):
            ex.pad_ssi_masks(max_dec_sents, max_enc_seq_len, 0.)

        self.ssi_masks_padded = np.stack([ex.ssi_masks_padded for ex in example_list])
        # self.ssi_masks_padded = np.ones([len(example_list), max_dec_sents, max_enc_seq_len], dtype=float)

        batch_sent_indices = []
        for ex_idx, ex in enumerate(example_list):
            ex.pad_sent_indices(hps.max_dec_steps, 0)
            new_sent_indices = []
            for sent_idx in ex.sent_indices:
                new_sent_indices.append([ex_idx, sent_idx])
            batch_sent_indices.append(new_sent_indices)
        self.batch_sent_indices = np.array(batch_sent_indices, dtype=np.int32)

    def store_orig_strings(self, example_list):
        """Store the original article and abstract strings in the Batch object"""
        self.original_articles = [ex.original_article for ex in example_list] # list of lists
        self.raw_article_sents = [ex.raw_article_sents for ex in example_list]
        self.tokenized_sents = [ex.tokenized_sents for ex in example_list]
        self.word_ids_sents = [ex.word_ids_sents for ex in example_list]
        self.original_abstracts = [ex.original_abstract for ex in example_list] # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] # list of list of lists
        self.all_original_abstracts_sents = [ex.all_original_abstract_sents for ex in example_list] # list of list of list of lists
        if example_list[0].ssi is not None:
            self.ssis = [ex.ssi for ex in example_list]
            self.ssi_masks = [ex.ssi_masks for ex in example_list]


class Batcher(object):
    """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

    BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold

    def __init__(self, data_path, vocab, hps, single_pass, cnn_500_dm_500=False, example_generator=None):
        """Initialize the batcher. Start threads that process the data into batches.

        Args:
            data_path: tf.Example filepattern.
            vocab: Vocabulary object
            hps: hyperparameters
            single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set). Otherwise generate random batches indefinitely (useful for training).
        """
        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._single_pass = single_pass
        self._cnn_500_dm_500 = cnn_500_dm_500
        self._example_generator = example_generator

        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1	# just one thread to batch examples
            self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._finished_reading = False # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 16 # num threads to fill example queue
            self._num_batch_q_threads = 4	# num threads to fill batch queue
            self._bucketing_cache_size = 100 # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()


    def next_batch(self):
        """Return a Batch from the batch queue.

        If mode='decode' or 'calc_features' then each batch contains a single example repeated beam_size-many times; this is necessary for beam search.

        Returns:
            batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
        """
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get() # get the next Batch
        return batch

    def fill_example_queue(self):
        """Reads data from file and processes into Examples which are then placed into the example queue."""

        if self._example_generator is None:
            input_gen = self.text_generator(
                data.example_generator(self._data_path, self._single_pass, self._cnn_500_dm_500, is_original=False))
        else:
            input_gen = self.text_generator(self._example_generator)
        if self._hps.pg_mmr and self._hps.ssi_data_path != '':  # if use pg_mmr and bert
            print (util.bcolors.OKGREEN + "Loading SSI from BERT at %s" % os.path.join(self._hps.ssi_data_path, 'ssi.pkl') + util.bcolors.ENDC)
            with open(os.path.join(self._hps.ssi_data_path, 'ssi.pkl')) as f:
                ssi_triple_list = pickle.load(f)
                # ssi_list = [ssi_triple[1] for ssi_triple in ssi_triple_list]
        else:
            ssi_triple_list = None
        counter = 0
        while True:
            try:
                (article,
                 abstracts, doc_indices_str, raw_article_sents, ssi) = next(input_gen)  # read the next example from file. article and abstract are both strings.
            except StopIteration:  # if there are no more examples:
                logging.info("The example generator for this example queue filling thread has exhausted data.")
                if self._single_pass:
                    logging.info(
                        "single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    if ssi_triple_list is not None and counter < len(ssi_triple_list):
                        raise Exception('Len of ssi list (%d) is greater than number of examples (%d)' % (len(ssi_triple_list), counter))
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")
            if ssi_triple_list is not None:
                if counter >= len(ssi_triple_list):
                    raise Exception('Len of ssi list (%d) is less than number of examples (>=%d)' % (len(ssi_triple_list), counter))
                ssi_length_extractive = ssi_triple_list[counter][2]
                ssi = ssi_triple_list[counter][1]
                ssi = ssi[:ssi_length_extractive]

            article = article
            abstracts = [abstract for abstract in abstracts]
            if type(doc_indices_str) != str:
                doc_indices_str = doc_indices_str
            raw_article_sents = [sent for sent in raw_article_sents]

            all_abstract_sentences = [[sent.strip() for sent in data.abstract2sents(
                abstract)] for abstract in abstracts]
            if len(all_abstract_sentences) != 0:
                abstract_sentences = all_abstract_sentences[0]
            else:
                abstract_sentences = []
            doc_indices = [int(idx) for idx in doc_indices_str.strip().split()]
            if self._hps.by_instance:   # if we are running iteratively on only instances (a singleton/pair + a summary sentence), not the whole article
                for abs_idx, abstract_sentence in enumerate(abstract_sentences):
                    inst_ssi = ssi[abs_idx]
                    if len(inst_ssi) == 0:
                        continue
                    inst_abstract_sentences = abstract_sentence
                    inst_raw_article_sents = util.reorder(raw_article_sents, inst_ssi)
                    inst_article = ' '.join([' '.join(util.process_sent(sent, whitespace=True)) for sent in inst_raw_article_sents])
                    inst_doc_indices = [0] * len(inst_article.split())

                    if len(inst_article) == 0:  # See https://github.com/abisee/pointer-generator/issues/1
                        logging.warning(
                            'Found an example with empty article text. Skipping it.\n*********************************************')
                    elif len(inst_article.strip().split()) < 3 and self._hps.skip_with_less_than_3:
                        print(
                            'Article has less than 3 tokens, so skipping\n*********************************************')
                    elif len(inst_abstract_sentences.strip().split()) < 3 and self._hps.skip_with_less_than_3:
                        print(
                            'Abstract has less than 3 tokens, so skipping\n*********************************************')
                    else:
                        inst_example = Example(inst_article, [inst_abstract_sentences], all_abstract_sentences, inst_doc_indices, inst_raw_article_sents, None, self._vocab, self._hps)
                        self._example_queue.put(inst_example)
            else:
                example = Example(article, abstract_sentences, all_abstract_sentences, doc_indices, raw_article_sents, ssi, self._vocab, self._hps)  # Process into an Example.
                self._example_queue.put(example)  # place the Example in the example queue.

            # print "example num", counter
            counter += 1

    def fill_batch_queue(self):
        """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.

        In decode mode, makes batches that each contain a single example repeated.
        """
        while True:

            # print 'hi'
            if self._hps.mode != 'decode' and self._hps.mode != 'calc_features':
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self._hps.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_len) # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self._hps.batch_size):
                    batches.append(inputs[i:i + self._hps.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:	# each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._hps, self._vocab))

            elif self._hps.mode == 'decode': # beam search decode mode
                ex = self._example_queue.get()
                batch = preprocess_batch(ex, self._hps.batch_size, self._hps, self._vocab)
                self._batch_queue.put(batch)
            else:   # calc features mode
                inputs = []
                for _ in range(self._hps.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                    # print "_ %d"%_
                # print "inputs len%d"%len(inputs)
                # Group the sorted Examples into batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self._hps.batch_size):
                    # print i
                    batches.append(inputs[i:i + self._hps.batch_size])

                # if not self._single_pass:
                #     shuffle(batches)
                for b in batches:	# each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._hps, self._vocab))



    def watch_threads(self):
        """Watch example queue and batch queue threads and restart if dead."""
        while True:
            time.sleep(60)
            for idx,t in enumerate(self._example_q_threads):
                if not t.is_alive(): # if the thread is dead
                    logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx,t in enumerate(self._batch_q_threads):
                if not t.is_alive(): # if the thread is dead
                    logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def text_generator(self, example_generator):
        """Generates article and abstract text from tf.Example.

        Args:
            example_generator: a generator of tf.Examples from file. See data.example_generator"""
        # i = 0

        while True:
            # i += 1
            e = next(example_generator) # e is a tf.Example
            abstract_texts = []
            raw_article_sents = []
            # if self._hps.pg_mmr or '_sent' in self._hps.dataset_name:
            try:
                # names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'), ('article_lcs_paths_list', 'delimited_list_of_list_of_lists')]
                # names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string_list'), ('corefs', 'json'), ('article_lcs_paths_list', 'delimited_list_of_list_of_lists')]
                names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string_list'), ('corefs', 'json')]
                if self._hps.dataset_name == 'duc_2004':
                    names_to_types[2] = ('summary_text', 'string_list')

                # raw_article_sents, ssi, groundtruth_summary_text, corefs, article_lcs_paths_list = util.unpack_tf_example(
                #     e, names_to_types)
                raw_article_sents, ssi, groundtruth_summary_sents, corefs = util.unpack_tf_example(
                    e, names_to_types)
                groundtruth_summary_text = '\n'.join(groundtruth_summary_sents)
                article_sent_tokens = [util.process_sent(sent) for sent in raw_article_sents]
                article_text = ' '.join([' '.join(sent) for sent in article_sent_tokens])
                if self._hps.dataset_name == 'duc_2004':
                    abstract_sentences = [['<s> ' + sent.strip() + ' </s>' for sent in
                                          gt_summ_text.strip().split('\n')] for gt_summ_text in groundtruth_summary_text]
                    abstract_sentences = [abs_sents[:max_dec_sents] for abs_sents in abstract_sentences]
                    abstract_texts = [' '.join(abs_sents) for abs_sents in abstract_sentences]
                else:
                    abstract_sentences = ['<s> ' + sent.strip() + ' </s>' for sent in groundtruth_summary_text.strip().split('\n')]
                    abstract_sentences = abstract_sentences[:max_dec_sents]
                    abstract_texts = [' '.join(abstract_sentences)]
                if 'doc_indices' not in e.features.feature or len(e.features.feature['doc_indices'].bytes_list.value) == 0:
                    num_words = len(article_text.split())
                    doc_indices_text = '0 ' * num_words
                else:
                    doc_indices_text = e.features.feature['doc_indices'].bytes_list.value[0]
                sentence_limit = 1 if self._hps.singles_and_pairs == 'singles' else 2
                ssi = util.enforce_sentence_limit(ssi, sentence_limit)
                ssi = ssi[:max_dec_sents]
                ssi = util.make_ssi_chronological(ssi)
            except:
                logging.error('Failed to get article or abstract from example')
                raise
            if len(article_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
                logging.warning('Found an example with empty article text. Skipping it.\n*********************************************')
            elif len(article_text.strip().split()) < 3 and self._hps.skip_with_less_than_3:
                print('Article has less than 3 tokens, so skipping\n*********************************************')
            elif len(abstract_texts[0].strip().split()) < 3 and self._hps.skip_with_less_than_3:
                print('Abstract has less than 3 tokens, so skipping\n*********************************************')
            else:
                # print i
                yield (article_text, abstract_texts, doc_indices_text, raw_article_sents, ssi)


def get_delimited_list_of_lists(example, name):
    text = get_string(example, name)
    return [[int(i) for i in (l.strip().split(' ') if l != '' else [])] for l in text.strip().split(';')]

def get_string(example, name):
    return example.features.feature[name].bytes_list.value[0]

def preprocess_example(article, groundtruth_summ_sents, doc_indices_str, raw_article_sents, hps, vocab):
    if len(groundtruth_summ_sents) != 0:
        abstract_sentences = groundtruth_summ_sents[0]
    else:
        abstract_sentences = []
    doc_indices = [int(idx) for idx in doc_indices_str.strip().split()]
    example = Example(article, abstract_sentences, groundtruth_summ_sents, doc_indices, raw_article_sents, None, vocab, hps)  # Process into an Example.
    return example

def preprocess_batch(ex, batch_size, hps, vocab):
    b = [ex for _ in range(batch_size)]
    batch = Batch(b, hps, vocab)
    return batch

def create_batch(article, groundtruth_summ_sents, doc_indices_str, raw_article_sents, batch_size, hps, vocab):
    ex = preprocess_example(article, groundtruth_summ_sents, doc_indices_str, raw_article_sents, hps, vocab)
    batch = preprocess_batch(ex, batch_size, hps, vocab)
    return batch