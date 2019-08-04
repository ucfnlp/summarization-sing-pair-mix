import collections

import util
from absl import app, flags
from tqdm import tqdm
import os
import glob
import data
import convert_data

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Makes a log dir based on name.\
                                               Must be one of {cnn_dm, xsum, duc_2004}')
flags.DEFINE_string('data_root', 'data/tf_data', 'Path to root directory for all datasets (already converted to TensorFlow examples).')
flags.DEFINE_string('dataset_split', 'all', 'Which dataset split to use. Must be one of {train, val, test, all}')


# names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json')]
names_to_types = [('raw_article_sents', 'string_list'), ('article', 'string'), ('abstract', 'string_list'), ('doc_indices', 'string')]
VOCAB_SIZE = 200000

def main(unused_argv):
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.dataset_split == 'all':
        dataset_splits = ['test', 'val', 'train']
    else:
        dataset_splits = [FLAGS.dataset_split]

    vocab_counter = collections.Counter()

    for dataset_split in dataset_splits:

        source_dir = os.path.join(FLAGS.data_root, FLAGS.dataset_name)
        source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))

        total = len(source_files) * 1000
        example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False,
                                                   should_check_valid=False)

        for example_idx, example in enumerate(tqdm(example_generator, total=total)):

            # raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs = util.unpack_tf_example(
            #     example, names_to_types)
            raw_article_sents, article, abstracts, doc_indices = util.unpack_tf_example(
                example, names_to_types)
            article_sent_tokens = [util.process_sent(sent) for sent in raw_article_sents]
            # groundtruth_summ_sent_tokens = [sent.strip().split() for sent in groundtruth_summary_text.strip().split('\n')]
            groundtruth_summ_sent_tokens = [[token for token in abstract.strip().split() if token not in ['<s>','</s>']] for abstract in abstracts]
            all_tokens = util.flatten_list_of_lists(article_sent_tokens) + util.flatten_list_of_lists(groundtruth_summ_sent_tokens)

            vocab_counter.update(all_tokens)

    print("Writing vocab file...")
    with open(os.path.join('logs', "vocab_" + FLAGS.dataset_name), 'w') as writer:
        for word, count in vocab_counter.most_common(VOCAB_SIZE):
            writer.write(word + ' ' + str(count) + '\n')
    print("Finished writing vocab file")





if __name__ == '__main__':
    app.run(main)


