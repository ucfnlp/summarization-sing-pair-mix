
from tqdm import tqdm
import copy
import os
import hashlib
import json

dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
out_dir = os.path.join('data/processed/cnn_dm')

all_train_urls = "data/cnn_dm_unprocessed/url_lists/all_train.txt"
all_val_urls = "data/cnn_dm_unprocessed/url_lists/all_val.txt"
all_test_urls = "data/cnn_dm_unprocessed/url_lists/all_test.txt"
cnn_tokenized_stories_dir = 'data/cnn_dm_unprocessed/cnn_stories_tokenized'
dm_tokenized_stories_dir = 'data/cnn_dm_unprocessed/dm_stories_tokenized'


# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506
num_expected_stories = {'test' : 11490,
                        'val' : 13367,
                        'train' : 287221}

def read_text_file(text_file):
  lines = []
  with open(text_file, "r", encoding='utf-8') as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s.encode())
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  return line + " ."

def move_from_quote_end_to_sent_end(tokens, cur_end_idx):
    spots_left_to_check = 5
    new_end_idx = cur_end_idx
    for i in range(1, spots_left_to_check+1):
        if cur_end_idx+i >= len(tokens):
            break
        if tokens[cur_end_idx+i] == '.':
            new_end_idx = cur_end_idx+i
            break
    return new_end_idx

def sent_tokenize_paragraph(tokens):
  sents = []
  while len(tokens) > 0:
      idx = next((i for i in range(len(tokens)) if (tokens[i] == '.' or tokens[i] == '?')), len(tokens)-1)
      if tokens[idx] == '?':
          is_part_of_quote = False
          if idx+1 < len(tokens) and tokens[idx+1] == "'":
              idx = idx + 1
              is_part_of_quote = True
          if idx+1 < len(tokens) and tokens[idx+1] == "''":
              idx = idx + 1
              is_part_of_quote = True
          if is_part_of_quote:
              idx = move_from_quote_end_to_sent_end(tokens, idx)
      else:
          if idx+1 < len(tokens) and tokens[idx+1] == "'":
              idx = idx + 1
          if idx+1 < len(tokens) and tokens[idx+1] == "''":
              idx = idx + 1
      sent = tokens[:idx+1]
      if len(sent) > 0:
          sents.append(' '.join(sent))
      tokens = tokens[idx+1:]
  return sents

def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_sents = []
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
      # Each line is a paragraph that contains 1 or more sentences
      tokens = line.split(' ')
      sents = sent_tokenize_paragraph(tokens)   # uses some heuristics to tokenize the paragraphs into sentences
      article_sents.extend(sents)

  return article_sents, highlights

def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))

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

def write_to_files(url_file, dataset_split):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print("Making bin file for URLs listed in %s..." % url_file)
  url_list = read_text_file(url_file)
  url_hashes = get_url_hashes(url_list)
  story_fnames = [s+".story" for s in url_hashes]
  num_stories = len(story_fnames)
  out_split_dir = os.path.join(out_dir, dataset_split)
  if not os.path.exists(out_split_dir):
      os.makedirs(out_split_dir)

  out_art = os.path.join(out_split_dir, 'articles.tsv')
  out_abs = os.path.join(out_split_dir, 'summaries.tsv')

  with open(out_art, 'wb') as f_art,\
      open(out_abs, 'wb') as f_abs:
    for idx,s in enumerate(tqdm(url_hashes)):
      if idx % 1000 == 0:
          print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))
      name = s
      story_fname = name + '.story'

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

      # Get the strings to write to .bin file
      article_sents, abstract_sents = get_art_abs(story_file)

      article_line = '\t'.join(article_sents) + '\n'
      abstract_line = '\t'.join(abstract_sents) + '\n'
      f_art.write(article_line.encode())
      f_abs.write(abstract_line.encode())


  print("Finished writing file %s\n" % url_file)


def main():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    write_to_files(all_test_urls, 'test')
    write_to_files(all_val_urls, 'val')
    write_to_files(all_train_urls, 'train')


if __name__ == '__main__':
    main()











































