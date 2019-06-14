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

"""This is the top-level file to test your summarization model"""

import os
import shutil
from distutils.dir_util import copy_tree

import tensorflow as tf
from collections import namedtuple
from data import Vocab
from batcher import Batcher
from model import SummarizationModel
from decode import BeamSearchDecoder
import convert_data
import importance_features
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import dill
from absl import app, flags, logging
import random
import util
import time
from tensorflow.python import debug as tf_debug
from tqdm import tqdm
import numpy as np
import glob

random.seed(222)
FLAGS = flags.FLAGS
original_pretrained_path = {'cnn_dm': 'logs/pretrained_model_tf1.2.1',
                            # 'xsum': 'logs/xsum',
                            'xsum': 'logs/xsum',
                            'duc_2004': 'logs/pretrained_model_tf1.2.1'
                            }

# Where to find data
flags.DEFINE_string('dataset_name', 'example_custom_dataset', 'Which dataset to use. Makes a log dir based on name.\
                                                Must be one of {tac_2011, tac_2008, duc_2004, duc_tac, cnn_dm} or a custom dataset name')
flags.DEFINE_string('data_root', os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi_and_tag_tokens', 'Path to root directory for all datasets (already converted to TensorFlow examples).')
flags.DEFINE_string('vocab_path', 'logs/vocab', 'Path expression to text vocabulary file.')
flags.DEFINE_string('pretrained_path', original_pretrained_path['cnn_dm'], 'Directory of pretrained model from See et al.')
flags.DEFINE_boolean('use_pretrained', False, 'If True, use pretrained model in the path FLAGS.pretrained_path.')


# Where to save output
flags.DEFINE_string('log_root', 'logs', 'Root directory for all logging.')
flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Don't change these settings
flags.DEFINE_string('mode', 'decode', 'must be one of train/eval/decode')
flags.DEFINE_boolean('single_pass', True, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
flags.DEFINE_string('actual_log_root', '', 'Dont use this setting, only for internal use. Root directory for all logging.')
flags.DEFINE_string('dataset_split', 'test', 'Which dataset split to use. Must be one of {train, val, test}')

# Hyperparameters
flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
flags.DEFINE_integer('batch_size', 16, 'minibatch size')
flags.DEFINE_integer('max_enc_steps', 100000, 'max timesteps of encoder (max source text tokens)')
flags.DEFINE_integer('max_dec_steps', 120, 'max timesteps of decoder (max summary tokens)')
flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
flags.DEFINE_integer('min_dec_steps', 100, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
flags.DEFINE_float('lr', 0.15, 'learning rate')
flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")

# PG-MMR settings
flags.DEFINE_boolean('pg_mmr', False, 'If true, use the PG-MMR model.')
flags.DEFINE_string('importance_fn', 'tfidf', 'Which model to use for calculating importance. Must be one of {svr, tfidf, oracle}.')
flags.DEFINE_float('lambda_val', 0.6, 'Lambda factor to reduce similarity amount to subtract from importance. Set to 0.5 to make importance and similarity have equal weight.')
flags.DEFINE_integer('mute_k', 7, 'Pick top k sentences to select and mute all others. Set to -1 to turn off.')
flags.DEFINE_boolean('retain_mmr_values', False, 'Only used if using mute mode. If true, then the mmr being\
                            multiplied by alpha will not be a 0/1 mask, but instead keeps their values.')
flags.DEFINE_string('similarity_fn', 'rouge_l', 'Which similarity function to use when calculating\
                            sentence similarity or coverage. Must be one of {rouge_l, ngram_similarity}')
flags.DEFINE_boolean('plot_distributions', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_integer('num_iterations', 60000, 'How many iterations to run. Set to -1 to run indefinitely.')

flags.DEFINE_boolean('attn_vis', False, 'If true, then output attention visualization during decoding.')
flags.DEFINE_boolean('lambdamart_input', True, 'If true, then do postprocessing to combine sentences from the same example.')
flags.DEFINE_string('singles_and_pairs', 'none',
                    'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both, none}.')
flags.DEFINE_string('original_dataset_name', '',
                    'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
flags.DEFINE_boolean('skip_with_less_than_3', True,
                    'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
flags.DEFINE_boolean('use_bert', True, 'If true, use PG trained on Websplit for testing.')
flags.DEFINE_boolean('upper_bound', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_string('ssi_data_path', '',
                    'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
flags.DEFINE_boolean('notrain', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('finetune', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
# flags.DEFINE_boolean('l_sents', True, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('word_imp_reg', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('convert_to_importance_model', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_float('imp_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')
flags.DEFINE_boolean('tag_tokens', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('by_instance', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')


flags.DEFINE_bool(
    "sentemb", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool("artemb", True, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("plushidden", True, "Whether to use TPU or GPU/CPU.")


kaiqiang_dataset_names = ['gigaword', 'cnndm_1to1', 'newsroom', 'websplit']

def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    """Calculate the running average loss via exponential decay.
    This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

    Args:
        loss: loss on the most recent eval step
        running_avg_loss: running_avg_loss so far
        summary_writer: FileWriter object to write for tensorboard
        step: training iteration step
        decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

    Returns:
        running_avg_loss: new running average loss
    """
    if running_avg_loss == 0:	# on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)	# clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss


def restore_best_model():
    """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
    logging.info("Restoring bestmodel for training...")

    # Initialize all vars in the model
    sess = tf.Session(config=util.get_config())
    print("Initializing all variables...")
    sess.run(tf.initialize_all_variables())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
    print("Restoring all non-adagrad variables from best model in eval dir...")
    curr_ckpt = util.load_ckpt(saver, sess, "eval")
    print("Restored %s." % curr_ckpt)

    # Save this model to train dir and quit
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
    print("Saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, new_fname)
    print("Saved.")
    exit()


def convert_to_coverage_model():
    """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
    logging.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=util.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
    print("restoring non-coverage variables...")
    curr_ckpt = util.load_ckpt(saver, sess)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()

def convert_to_importance_model():
    """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
    logging.info("converting non-importance model to importance model..")

    new_log_root = FLAGS.log_root + '_imp' + str(FLAGS.imp_loss_wt)

    print("copying models from %s to %s..." % (FLAGS.log_root, new_log_root))
    util.create_dirs(new_log_root)
    copy_tree(FLAGS.log_root, new_log_root)
    print("copied.")
    # exit()

def setup_training(model, batcher):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train")
    if FLAGS.finetune:
        if not os.path.exists(train_dir):
            print (util.bcolors.OKGREEN + 'Copying See et al. pre-trained model (%s) to (%s) to be fine-tuned' % (os.path.join(FLAGS.pretrained_path, 'train'), train_dir) + util.bcolors.ENDC)
            os.makedirs(train_dir)
            files = glob.glob(os.path.join(os.path.join(FLAGS.pretrained_path, 'train'), "*model*"))
            files.extend(glob.glob(os.path.join(os.path.join(FLAGS.pretrained_path, 'train'), "*checkpoint*")))
            for file in files:
                if os.path.isfile(file):
                    shutil.copy2(file, train_dir)
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    model.build_graph() # build the graph
    if FLAGS.convert_to_coverage_model:
        assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
        convert_to_coverage_model()
    if FLAGS.restore_best_model:
        restore_best_model()
    saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time

    sv = tf.train.Supervisor(logdir=train_dir,
                                         is_chief=True,
                                         saver=saver,
                                         summary_op=None,
                                         save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                                         save_model_secs=60, # checkpoint every 60 secs
                                         global_step=model.global_step)
    summary_writer = sv.summary_writer
    logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
    logging.info("Created session.")
    try:
        run_training(model, batcher, sess_context_manager, sv, summary_writer) # this is an infinite loop until interrupted
    except KeyboardInterrupt:
        logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        sv.stop()


def run_training(model, batcher, sess_context_manager, sv, summary_writer):
    """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
    logging.info("starting run_training")
    with sess_context_manager as sess:
        if FLAGS.debug: # start the tensorflow debugger
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        if FLAGS.num_iterations == -1:
            while True: # repeats until interrupted
                run_training_iteration(model, batcher, summary_writer, sess)
        else:
            initial_iter = model.global_step.eval(sess)
            pbar = tqdm(initial=initial_iter, total=FLAGS.num_iterations)
            print(("Starting at iteration %d" % initial_iter))
            for iter_idx in range(initial_iter, FLAGS.num_iterations):
                run_training_iteration(model, batcher, summary_writer, sess)
                pbar.update(1)
            pbar.close()

def run_training_iteration(model, batcher, summary_writer, sess):
    batch = batcher.next_batch()

    # tqdm.write('running training step...')
    t0=time.time()
    results = model.run_train_step(sess, batch)
    t1=time.time()
    # tqdm.write('seconds for training step: %.3f' % (t1-t0))

    loss = results['loss']
    tqdm.write('loss: %f' % loss) # print the loss to screen

    if not np.isfinite(loss):
        raise util.InfinityValueError("Loss is not finite. Stopping.")

    if FLAGS.coverage:
        coverage_loss = results['coverage_loss']
        tqdm.write("coverage_loss: %f" % coverage_loss) # print the coverage loss to screen

    if FLAGS.word_imp_reg:
        importance_loss = results['importance_loss']
        tqdm.write("importance_loss: %f" % importance_loss)  # print the coverage loss to screen

    # get the summaries and iteration number so we can write summaries to tensorboard
    summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
    train_step = results['global_step'] # we need this to update our running average loss

    summary_writer.add_summary(summaries, train_step) # write the summaries
    if train_step % 100 == 0: # flush the summary writer every so often
        summary_writer.flush()

def run_eval(model, batcher, vocab):
    """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
    model.build_graph() # build the graph
    saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
    sess = tf.Session(config=util.get_config())
    eval_dir = os.path.join(FLAGS.log_root, "eval") # make a subdir of the root dir for eval data
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
    summary_writer = tf.summary.FileWriter(eval_dir)
    running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    best_loss = None	# will hold the best loss achieved so far

    while True:
        _ = util.load_ckpt(saver, sess) # load a new checkpoint
        batch = batcher.next_batch() # get the next batch

        # run eval on the batch
        t0=time.time()
        results = model.run_eval_step(sess, batch)
        t1=time.time()
        logging.info('seconds for batch: %.2f', t1-t0)

        # print the loss and coverage loss to screen
        loss = results['loss']
        logging.info('loss: %f', loss)
        if FLAGS.coverage:
            coverage_loss = results['coverage_loss']
            logging.info("coverage_loss: %f", coverage_loss)

        # add summaries
        summaries = results['summaries']
        train_step = results['global_step']
        summary_writer.add_summary(summaries, train_step)

        # calculate running avg loss
        running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)

        # If running_avg_loss is best so far, save this checkpoint (early stopping).
        # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
        if best_loss is None or running_avg_loss < best_loss:
            logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
            saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
            best_loss = running_avg_loss

        # flush the summary writer every so often
        if train_step % 100 == 0:
            summary_writer.flush()

def calc_features(cnn_dm_train_data_path, hps, vocab, batcher, save_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    decode_model_hps = hps  # This will be the hyperparameters for the decoder model
    model = SummarizationModel(decode_model_hps, vocab)
    decoder = BeamSearchDecoder(model, batcher, vocab)
    decoder.calc_importance_features(cnn_dm_train_data_path, hps, save_path, 1000)

def fit_tfidf_vectorizer(hps, vocab):
    if not os.path.exists(os.path.join(FLAGS.actual_log_root, 'tfidf_vectorizer')):
        os.makedirs(os.path.join(FLAGS.actual_log_root, 'tfidf_vectorizer'))

    decode_model_hps = hps._replace(max_dec_steps=1, batch_size=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries

    batcher = Batcher(FLAGS.data_path, vocab, decode_model_hps, single_pass=FLAGS.single_pass)
    all_sentences = []
    while True:
        batch = batcher.next_batch()	# 1 example repeated across batch
        if batch is None: # finished decoding dataset in single_pass mode
            break
        all_sentences.extend(batch.raw_article_sents[0])

    stemmer = PorterStemmer()

    class StemmedTfidfVectorizer(TfidfVectorizer):
        def build_analyzer(self):
            analyzer = super(TfidfVectorizer, self).build_analyzer()
            return lambda doc: (stemmer.stem(w) for w in analyzer(doc))

    tfidf_vectorizer = StemmedTfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 3), max_df=0.7)
    tfidf_vectorizer.fit_transform(all_sentences)
    return tfidf_vectorizer

log_dir = 'logs'

def main(unused_argv):
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    # if '_sent' in FLAGS.dataset_name:
    #     FLAGS.data_root = os.path.expanduser('~') + '/data/tf_data/with_coref_and_tag_tokens'
    if FLAGS.pg_mmr:
        FLAGS.data_root = os.path.expanduser('~') + "/data/tf_data/with_coref_and_ssi"
    if FLAGS.dataset_name != "":
        FLAGS.data_path = os.path.join(FLAGS.data_root, FLAGS.dataset_name, FLAGS.dataset_split + '*')
    if FLAGS.dataset_name in kaiqiang_dataset_names:
        FLAGS.skip_with_less_than_3 = False
    if not os.path.exists(os.path.join(FLAGS.data_root, FLAGS.dataset_name)) or len(os.listdir(os.path.join(FLAGS.data_root, FLAGS.dataset_name))) == 0:
        print(('No TF example data found at %s so creating it from raw data.' % os.path.join(FLAGS.data_root, FLAGS.dataset_name)))
        convert_data.process_dataset(FLAGS.dataset_name)

    if FLAGS.mode == 'decode':
        extractor = '_bert' if FLAGS.use_bert else '_lambdamart'
        FLAGS.use_pretrained = True
        FLAGS.single_pass = True
    else:
        extractor = ''
    pretrained_dataset = FLAGS.dataset_name
    if FLAGS.dataset_name == 'duc_2004':
        pretrained_dataset = 'cnn_dm'
    if FLAGS.pg_mmr:
        FLAGS.exp_name += '_pgmmr'
    if FLAGS.singles_and_pairs == 'both':
        FLAGS.exp_name = FLAGS.exp_name + extractor + '_both'
        if FLAGS.mode == 'decode':
            FLAGS.pretrained_path = os.path.join(FLAGS.log_root, pretrained_dataset + '_pgmmr_both')
        dataset_articles = FLAGS.dataset_name
    elif FLAGS.singles_and_pairs == 'singles':
        FLAGS.exp_name = FLAGS.exp_name + extractor + '_singles'
        if FLAGS.mode == 'decode':
            FLAGS.pretrained_path = os.path.join(FLAGS.log_root, pretrained_dataset + '_pgmmr_singles')
        dataset_articles = FLAGS.dataset_name + '_singles'

    if FLAGS.notrain:
        FLAGS.exp_name += '_notrain'
        FLAGS.pretrained_path = original_pretrained_path[FLAGS.dataset_name]
    if FLAGS.finetune:
        FLAGS.exp_name += '_finetune'
        if FLAGS.mode == 'decode':
            FLAGS.pretrained_path += '_finetune'

    extractor = 'bert' if FLAGS.use_bert else 'lambdamart'
    bert_suffix = ''
    if FLAGS.use_bert:
        if FLAGS.sentemb:
            bert_suffix += '_sentemb'
        if FLAGS.artemb:
            bert_suffix += '_artemb'
        if FLAGS.plushidden:
            bert_suffix += '_plushidden'
        # if FLAGS.mode == 'decode':
        #     if FLAGS.sentemb:
        #         FLAGS.exp_name += '_sentemb'
        #     if FLAGS.artemb:
        #         FLAGS.exp_name += '_artemb'
        #     if FLAGS.plushidden:
        #         FLAGS.exp_name += '_plushidden'
    if FLAGS.upper_bound:
        FLAGS.exp_name = FLAGS.exp_name + '_upperbound'
        ssi_list = None     # this is if we are doing the upper bound evaluation (ssi_list comes straight from the groundtruth)
    else:
        if FLAGS.mode == 'decode':
            my_log_dir = os.path.join(log_dir, '%s_%s_%s%s' % (FLAGS.dataset_name, extractor, FLAGS.singles_and_pairs, bert_suffix))
            FLAGS.ssi_data_path = my_log_dir

    logging.set_verbosity(logging.INFO) # choose what level of logging you want
    logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.exp_name = FLAGS.exp_name if FLAGS.exp_name != '' else FLAGS.dataset_name
    FLAGS.actual_log_root = FLAGS.log_root
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)

    if FLAGS.convert_to_importance_model:
        convert_to_importance_model()
        FLAGS.convert_to_coverage_model = True
    if FLAGS.word_imp_reg:
        assert FLAGS.coverage, "To run with importance_loss, run with coverage=True as well"
        FLAGS.log_root += '_imp' + str(FLAGS.imp_loss_wt)
    if FLAGS.tag_tokens:
        FLAGS.log_root += '_tag'

    print(util.bcolors.OKGREEN + "Experiment path: " + FLAGS.log_root + util.bcolors.ENDC)

    if FLAGS.dataset_name == 'duc_2004':
        vocab = Vocab(FLAGS.vocab_path + '_' + 'cnn_dm', FLAGS.vocab_size) # create a vocabulary
    else:
        vocab_datasets = [os.path.basename(file_path).split('vocab_')[1] for file_path in glob.glob(FLAGS.vocab_path + '_*')]
        original_dataset_name = [file_name for file_name in vocab_datasets if file_name in FLAGS.dataset_name]
        if len(original_dataset_name) > 1:
            raise Exception('Too many choices for vocab file')
        if len(original_dataset_name) < 1:
            raise Exception('No vocab file for dataset created. Run make_vocab.py --dataset_name=<my original dataset name>')
        original_dataset_name = original_dataset_name[0]
        FLAGS.original_dataset_name = original_dataset_name
        vocab = Vocab(FLAGS.vocab_path + '_' + original_dataset_name, FLAGS.vocab_size) # create a vocabulary


    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size

    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and FLAGS.mode!='decode':
        raise Exception("The single_pass flag should only be True in decode mode")

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    # hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std',
    #                'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps',
    #                'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen', 'lambdamart_input', 'pg_mmr', 'singles_and_pairs', 'skip_with_less_than_3', 'ssi_data_path',
    #                'dataset_name', 'word_imp_reg', 'imp_loss_wt', 'tag_tokens']
    hparam_list = [item for item in list(FLAGS.flag_values_dict().keys()) if item != '?']
    hps_dict = {}
    for key,val in FLAGS.__flags.items(): # for each flag
        if key in hparam_list: # if it's in the list
            hps_dict[key] = val.value # add it to the dict
    hps = namedtuple("HParams", list(hps_dict.keys()))(**hps_dict)

    if FLAGS.pg_mmr:

        # Fit the TFIDF vectorizer if not already fitted
        if FLAGS.importance_fn == 'tfidf':
            tfidf_model_path = os.path.join(FLAGS.actual_log_root, 'tfidf_vectorizer', FLAGS.original_dataset_name + '.dill')
            if not os.path.exists(tfidf_model_path):
                print(('No TFIDF vectorizer model file found at %s, so fitting the model now.' % tfidf_model_path))
                tfidf_vectorizer = fit_tfidf_vectorizer(hps, vocab)
                with open(tfidf_model_path, 'wb') as f:
                    dill.dump(tfidf_vectorizer, f)

        # Train the SVR model on the CNN validation set if not already trained
        if FLAGS.importance_fn == 'svr':
            save_path = os.path.join(FLAGS.data_root, 'svr_training_data')
            importance_model_path = os.path.join(FLAGS.actual_log_root, 'svr.pickle')
            dataset_split = 'val'
            if not os.path.exists(importance_model_path):
                if not os.path.exists(save_path) or len(os.listdir(save_path)) == 0:
                    print(('No importance_feature instances found at %s so creating it from raw data.' % save_path))
                    decode_model_hps = hps._replace(
                        max_dec_steps=1, batch_size=100, mode='calc_features')  # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
                    cnn_dm_train_data_path = os.path.join(FLAGS.data_root, 'cnn_500_dm_500', dataset_split + '*')
                    batcher = Batcher(cnn_dm_train_data_path, vocab, decode_model_hps, single_pass=FLAGS.single_pass, cnn_500_dm_500=True)
                    calc_features(cnn_dm_train_data_path, decode_model_hps, vocab, batcher, save_path)

                print(('No importance_feature SVR model found at %s so training it now.' % importance_model_path))
                features_list = importance_features.get_features_list(True)
                sent_reps = importance_features.load_data(os.path.join(save_path, dataset_split + '*'), -1)
                print('Loaded %d sentences representations' % len(sent_reps))
                x_y = importance_features.features_to_array(sent_reps, features_list)
                train_x, train_y = x_y[:,:-1], x_y[:,-1]
                svr_model = importance_features.run_training(train_x, train_y)
                with open(importance_model_path, 'wb') as f:
                    pickle.dump(svr_model, f)

    # Create a batcher object that will create minibatches of data
    batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    tf.set_random_seed(113) # a seed value for randomness

    # Start decoding on multi-document inputs
    if hps.mode == 'train':
        print("creating model...")
        model = SummarizationModel(hps, vocab)
        setup_training(model, batcher)
    elif hps.mode == 'eval':
        model = SummarizationModel(hps, vocab)
        run_eval(model, batcher, vocab)
    elif hps.mode == 'decode':
        decode_model_hps = hps._replace(max_dec_steps=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
        model = SummarizationModel(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(model, batcher, vocab)
        decoder.decode() # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
        # while True:
        #     a=0
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")

if __name__ == '__main__':
    app.run(main)
