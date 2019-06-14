import os
from tqdm import tqdm
from absl import flags
from absl import app
import util
import sys

FLAGS = flags.FLAGS

if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
if 'dataset_split' not in flags.FLAGS:
    flags.DEFINE_string('dataset_split', 'test', 'Which dataset split to use. Must be one of {train, val (or dev), test}')
if 'sentence_limit' not in flags.FLAGS:
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
if 'num_instances' not in flags.FLAGS:
    flags.DEFINE_integer('num_instances', 10,
                         'Number of instances to run for before stopping. Use -1 to run on all instances.')

FLAGS(sys.argv)

from ssi_functions import get_simple_source_indices_list, html_highlight_sents_in_article
from ssi_functions import write_highlighted_html

data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi'

ssi_dir = 'data/ssi'
highlight_root = 'data/correctness/highlighted'
processed_root = 'data/correctness/processed'
systems = ['reference', 'novel', 'dca', 'abs-rl-rerank', 'pg', 'bottom-up']
# systems = ['novel', 'dca']
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'), ('doc_indices', 'delimited_list')]
min_matched_tokens = 2



def main(unused_argv):

    print('Running statistics on %s' % FLAGS.dataset_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    util.create_dirs(highlight_root)

    stats = {}
    for system in systems:
        print('Processing ' + system + '...')
        num_compress = 0
        num_fuse = 0
        num_copy = 0
        num_fail = 0
        highlight_dir = os.path.join(highlight_root, system)
        processed_dir = os.path.join(processed_root, system)
        util.create_dirs(highlight_dir)

        f_ssi = open(os.path.join(processed_dir, 'source_indices.txt'), 'w')
        f_summ = open(os.path.join(processed_dir, 'summaries.txt'))
        f_article = open(os.path.join(processed_root, 'article', 'articles.txt'))

        for example_idx in tqdm(range(11490)):
            if FLAGS.num_instances != -1 and example_idx >= FLAGS.num_instances:
                break
            summary_sent_tokens = [sent.split() for sent in f_summ.readline().strip().split('\t')]
            article_sent_tokens = [sent.split() for sent in f_article.readline().lower().strip().split('\t')]

            groundtruth_ssi_list, lcs_paths_list, article_lcs_paths_list = get_simple_source_indices_list(
                summary_sent_tokens,
                article_sent_tokens, None, FLAGS.sentence_limit, min_matched_tokens)
            groundtruth_highlighted_html = html_highlight_sents_in_article(summary_sent_tokens,
                                                                           groundtruth_ssi_list,
                                                                           article_sent_tokens,
                                                                           lcs_paths_list=lcs_paths_list,
                                                                           article_lcs_paths_list=article_lcs_paths_list)
            all_html = '<u>System Summary</u><br><br>' + groundtruth_highlighted_html
            write_highlighted_html(all_html, highlight_dir, example_idx)
            f_ssi.write('\t'.join([','.join(str(idx) for idx in source_indices) if len(source_indices) >= 1 else '-1' for source_indices in groundtruth_ssi_list]) + '\n')
            for ssi_idx, ssi in enumerate(groundtruth_ssi_list):
                if len(ssi) >= 2:
                    num_fuse += 1
                elif len(ssi) == 1:
                    source_sent = ' '.join(article_sent_tokens[ssi[0]])
                    summ_sent = ' '.join(summary_sent_tokens[ssi_idx])
                    if source_sent == summ_sent:
                        num_copy += 1
                    else:
                        num_compress += 1
                        # tqdm.write(source_sent + '\n' + summ_sent + '\n\n')
                else:
                    num_fail += 1
            a=0
        stats[system] = (num_compress, num_fuse, num_copy, num_fail)
        f_summ.close()
        f_article.close()
        f_ssi.close()
    print("num_compress, num_fuse, num_copy, num_fail")
    for system in systems:
        print(system)
        total = sum(stats[system]) * 1.
        print('\t'.join(["%.2f" % (val*100/total) for val in stats[system]]))








if __name__ == '__main__':
    app.run(main)



