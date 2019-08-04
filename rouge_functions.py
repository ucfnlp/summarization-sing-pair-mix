
import collections
import six
import pyrouge
import logging as log
import os
import sys
from absl import flags
FLAGS = flags.FLAGS
try:
    reload(sys)
    sys.setdefaultencoding('utf8')
except:
    _=None
import tempfile
tempfile.tempdir = "tmp"
if not os.path.exists(tempfile.tempdir):
    os.makedirs(tempfile.tempdir)

def _ngrams(words, n):
    queue = collections.deque(maxlen=n)
    for w in words:
        queue.append(w)
        if len(queue) == n:
            yield tuple(queue)

def _ngram_counts(words, n):
    return collections.Counter(_ngrams(words, n))

def _ngram_count(words, n):
    return max(len(words) - n + 1, 0)

def _counter_overlap(counter1, counter2):
    result = 0
    for k, v in six.iteritems(counter1):
        result += min(v, counter2[k])
    return result

def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0

def _safe_f1(matches, recall_total, precision_total, alpha):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score + alpha * recall_score
    if denom > 0.0:
        return (precision_score * recall_score) / denom
    else:
        return 0.0

def rouge_n(peer, models, n, alpha, metric='f1'):
    """
    Compute the ROUGE-N score of a peer with respect to one or more models, for
    a given value of `n`.
    """
    if len(models) == 0:
        return 0.

    if type(models[0]) is not list:
        models = [models]

    matches = 0
    recall_total = 0
    peer_counter = _ngram_counts(peer, n)
    for model in models:
        model_counter = _ngram_counts(model, n)
        matches += _counter_overlap(peer_counter, model_counter)
        recall_total += _ngram_count(model, n)
    precision_total = len(models) * _ngram_count(peer, n)
    if metric == 'f1':
        return _safe_f1(matches, recall_total, precision_total, alpha)
    elif metric == 'precision':
        return _safe_divide(matches, precision_total)
    elif metric == 'recall':
        return _safe_divide(matches, recall_total)
    else:
        raise Exception('must be one of {f1, recall, precision}')

def rouge_1(peer, models, alpha, metric='f1'):
    """
    Compute the ROUGE-1 (unigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 1, alpha, metric='f1')

def rouge_2(peer, models, alpha, metric='f1'):
    """
    Compute the ROUGE-2 (bigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 2, alpha, metric='f1')




def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s

def write_for_rouge(all_reference_sents, decoded_sents, ex_index, ref_dir, dec_dir, decoded_words=None, file_name=None, log=True):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
        all_reference_sents: list of list of strings
        decoded_sents: list of strings
        ex_index: int, the index with which to label the files
    """

    # First, divide decoded output into sentences if we supply words instead of sentences
    if decoded_words is not None:
        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index(".")
            except ValueError:  # there is text remaining that doesn't end in "."
                fst_period_idx = len(decoded_words)
            sent = decoded_words[:fst_period_idx + 1]  # sentence up to and including the period
            decoded_words = decoded_words[fst_period_idx + 1:]  # everything else
            decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    all_reference_sents = [[make_html_safe(w) for w in abstract] for abstract in all_reference_sents]

    # Write to file
    if file_name is None:
        decoded_file = os.path.join(dec_dir, "%06d_decoded.txt" % ex_index)
    else:
        decoded_file = os.path.join(dec_dir, "%s_%06d.txt" % (file_name, ex_index))

    for abs_idx, abs in enumerate(all_reference_sents):
        if file_name is None:
            ref_file = os.path.join(ref_dir, "%06d_reference.%s.txt" % (
                ex_index, chr(ord('A') + abs_idx)))
        else:
            ref_file = os.path.join(ref_dir, "%s_%06d.%s.txt" % (
                file_name, ex_index, chr(ord('A') + abs_idx)))
        with open(ref_file, "w") as f:
            for idx, sent in enumerate(abs):
                f.write(sent + "\n")
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(decoded_sents):
            f.write(sent + "\n")

def rouge_eval(ref_dir, dec_dir, l_param=100):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
#   r.model_filename_pattern = '#ID#_reference.txt'
    r.model_filename_pattern = '#ID#_reference.[A-Z].txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    log.getLogger('global').setLevel(log.WARNING) # silence pyrouge logging
    rouge_args = ['-e', r._data_dir,
         '-c',
         '95',
         '-2', '4',        # This is the only one we changed (changed the max skip from -1 to 4)
         '-U',
         '-r', '1000',
         '-n', '4',
         '-w', '1.2',
         '-a',
         '-l', str(l_param)]
    rouge_args = ' '.join(rouge_args)
    rouge_results = r.convert_and_evaluate(rouge_args=rouge_args)
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write, prefix=None, suffix=None):
    """Log ROUGE results to screen and write to file.

    Args:
        results_dict: the dictionary returned by pyrouge
        dir_to_write: the directory where we will write the results to"""
    log_str = ""
    for x in ["1","2","l","s4","su4"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x,y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str) # log to screen
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print(("Writing final ROUGE results to %s...", results_file))
    with open(results_file, "w") as f:
        f.write(log_str)

    print("\nROUGE-1, ROUGE-2, ROUGE-SU4 (PRF):\n")
    sheets_str = ""
    last_rouge_metric = "su4" if FLAGS.dataset_name == 'duc_2004' else "l"
    for x in ["1", "2", last_rouge_metric]:
        for y in ["precision", "recall", "f_score"]:
            key = "rouge_%s_%s" % (x, y)
            val = results_dict[key] * 100
            sheets_str += "%.2f\t" % (val)
    sheets_str += "\n"
    if prefix is not None:
        sheets_str = prefix + sheets_str
    if suffix is not None:
        sheets_str = sheets_str + suffix
    print(sheets_str)
    sheets_results_file = os.path.join(dir_to_write, "sheets_results.txt")
    with open(sheets_results_file, "w") as f:
        f.write(sheets_str)
    return sheets_str






















