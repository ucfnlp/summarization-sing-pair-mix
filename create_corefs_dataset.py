from tensorflow.core.example import example_pb2
import numpy as np
from tqdm import tqdm
import copy
import util
from absl import flags
from absl import app
import sys
import os
import hashlib
import struct
import subprocess
import collections
import json
import shutil

dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data
out_full_dir = os.path.join(os.path.expanduser('~') + '/data/tf_data/with_coref/cnn_dm/all')

all_train_urls = os.path.expanduser('~') + "/data/url_lists/all_train.txt"
all_val_urls = os.path.expanduser('~') + "/data/url_lists/all_val.txt"
all_test_urls = os.path.expanduser('~') + "/data/url_lists/all_test.txt"

cnn_tokenized_stories_dir = os.path.expanduser('~') + '/data/cnn_stories_tokenized'
dm_tokenized_stories_dir = os.path.expanduser('~') + '/data/dm_stories_tokenized'
corefs_dir = os.path.expanduser('~') + '/data/corenlp_corefs/processed/cnn_dm'

out_dir = os.path.expanduser('~') + '/data/tf_data/with_coref/cnn_dm'

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506
num_expected_stories = {'test' : 11490,
                        'val' : 13367,
                        'train' : 287221}

def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s)
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  # lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

  return article, abstract


def chunk_file(set_name, out_full_dir, out_dir):
  in_file = os.path.join(out_full_dir, '%s.bin' % set_name)
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(out_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1

def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))

def chunk_all(out_full_dir, out_dir):
  # Make a dir to hold the chunks
  if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name, out_full_dir, out_dir)
  print("Saved chunked data in %s" % out_dir)

def get_corefs(coref_file):
    with open(coref_file) as f:
        data = json.load(f)
    coref_ids = list(data['corefs'].keys())
    coref_list = [data['corefs'][id] for id in coref_ids]
    return coref_list

def get_sent_tokens(sent):
    return [token['originalText'] for token in sent['tokens']]

def get_tokenized_article(coref_file):
    with open(coref_file) as f:
        data = json.load(f)
    sentences = data['sentences']
    article_sent_tokens = [get_sent_tokens(sent) for sent in sentences]
    return article_sent_tokens

def fix_article_sent_tokens(article_sent_tokens):
    fixed = []
    for sent_tokens in article_sent_tokens:
        sent = []
        for item in sent_tokens:
            if type(item) == list:
                sent.extend(item)
            elif isinstance(item, str):
                sent.append(item)
            else:
                raise Exception('Item is not a string or a list: ' + str(item))
        fixed.append(sent)
    return fixed

    fixed_article_sent_tokens = fix_article_sent_tokens(coref_article_sent_tokens)
    return fixed_article_sent_tokens, corefs_skipped, total_corefs

def fix_trailing_apostrophe_s(corefs):
    fixed_corefs = copy.deepcopy(corefs)
    for mentions in fixed_corefs:
        for m in mentions:
            tokens = m['text'].split(' ')
            if len(tokens) == 1 and tokens[0] == "'s":
                tqdm.write('Warning: there was a mention that only contained " \'s ", so leaving this mention alone.')
            if tokens[-1] == "'s" or tokens[-1] == "'":
                m['text'] = ' '.join(tokens[:-1])
                m['endIndex'] = m['endIndex'] - 1
    return fixed_corefs

def remove_irrelevant(corefs):
    if len(corefs) == 0:
        return corefs
    relevant_keys = ['endIndex', 'isRepresentativeMention', 'sentNum', 'startIndex', 'text', 'type']
    irrelevant_keys = [k for k in list(corefs[0][0].keys()) if k not in relevant_keys]
    for mentions in corefs:
        for m in mentions:
            for key in irrelevant_keys:
                del m[key]
    return corefs

def write_to_bin(url_file, out_dir, dataset_split):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print("Making bin file for URLs listed in %s..." % url_file)
  url_list = read_text_file(url_file)
  url_hashes = get_url_hashes(url_list)
  file_names = os.listdir(os.path.join(corefs_dir, dataset_split))
  story_fnames = [s+".story" for s in url_hashes]
  num_stories = len(story_fnames)
  corenlp_paths = []
  corefs_split_dir = os.path.join(corefs_dir, dataset_split)
  corefs_skipped_list = []
  percent_corefs_skipped_list = []
  out_file = os.path.join(out_full_dir, dataset_split + '.bin')

  with open(out_file, 'wb') as writer:
    for idx,s in enumerate(tqdm(url_hashes)):
      if idx % 100 == 0:
          tqdm.write('Average corefs skipped: %.2f' % np.mean(corefs_skipped_list))
          tqdm.write('Average percent corefs skipped: %.2f' % np.mean(percent_corefs_skipped_list))
      if idx % 1000 == 0:
          print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))
      name = s
      story_fname = name + '.story'
      coref_fname = name + '.article.json'

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, story_fname)):
        story_file = os.path.join(cnn_tokenized_stories_dir, story_fname)
      elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, story_fname)):
        story_file = os.path.join(dm_tokenized_stories_dir, story_fname)
      else:
        print("Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?" % (story_fname, cnn_tokenized_stories_dir, dm_tokenized_stories_dir))
        # Check again if tokenized stories directories contain correct number of files
        print("Checking that the tokenized stories directories %s and %s contain correct number of files..." % (cnn_tokenized_stories_dir, dm_tokenized_stories_dir))
        check_num_stories(cnn_tokenized_stories_dir, num_expected_cnn_stories)
        check_num_stories(dm_tokenized_stories_dir, num_expected_dm_stories)
        raise Exception("Tokenized stories directories %s and %s contain correct number of files but story file %s found in neither." % (cnn_tokenized_stories_dir, dm_tokenized_stories_dir, story_fname))

      # Look in the corenlp parsed/coref dirs to find the .article.json file corresponding to this url
      if os.path.isfile(os.path.join(corefs_split_dir, coref_fname)):
        coref_file = os.path.join(corefs_split_dir, coref_fname)
      else:
        print("Error: Couldn't find coref file %s in either coref directory %s. Was there an error during preprocessing?" % (coref_fname, corefs_split_dir))
        # Check again if tokenized stories directories contain correct number of files
        print("Checking that the tokenized stories directory %s contain correct number of files..." % (corefs_split_dir))
        check_num_stories(corefs_split_dir, num_expected_stories[dataset_split])
        raise Exception("Tokenized stories directory %s contain correct number of files but story file %s found in neither." % (corefs_split_dir, coref_fname))

      # Get the strings to write to .bin file
      article, abstract = get_art_abs(story_file)
      article_sent_tokens = get_tokenized_article(coref_file)
      corefs = get_corefs(coref_file)
      fixed_corefs = fix_trailing_apostrophe_s(corefs)

      corefs_relevant_info = remove_irrelevant(fixed_corefs)
      corefs_json = json.dumps(corefs_relevant_info)

      raw_article_sents = [' '.join(sent).strip() for sent in article_sent_tokens]
      article_text = ' '.join(raw_article_sents)
      article_text = article_text.lower()     # Because we didn't lowercase in preprocess_for_coref.py
      abstract = abstract.lower()   # Because we didn't lowercase in get_art_abs

      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([article_text])
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])

      num_tokens_total = sum([len(sent) for sent in article_sent_tokens])
      doc_indices = ' '.join(['0'] * num_tokens_total)
      tf_example.features.feature['doc_indices'].bytes_list.value.extend([doc_indices])
      for sent in raw_article_sents:
        tf_example.features.feature['raw_article_sents'].bytes_list.value.extend([sent])
      tf_example.features.feature['corefs'].bytes_list.value.extend([corefs_json])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

  print("Finished writing file %s\n" % url_file)


def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    util.create_dirs(out_full_dir)
    util.create_dirs(out_dir)

    write_to_bin(all_test_urls, out_dir, 'test')
    write_to_bin(all_val_urls, out_dir, 'val')
    write_to_bin(all_train_urls, out_dir, 'train')

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all(out_full_dir, out_dir)
    shutil.rmtree(out_full_dir)


if __name__ == '__main__':
    app.run(main)











































