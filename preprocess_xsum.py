import util
import os
from tqdm import tqdm
import glob
import json

split_dict = json.loads(open(os.path.expanduser('~') + "/xsum/XSum/XSum-Dataset/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json").read())
data_types = ["test", "validation", "train"]

article_dir = os.path.expanduser('~') + '/xsum/XSum/XSum-Dataset/xsum-preprocessed/document'
summary_dir = os.path.expanduser('~') + '/xsum/XSum/XSum-Dataset/xsum-preprocessed/summary'
out_dir = os.path.expanduser('~') + '/xsum/xsum-logan'
util.create_dirs(out_dir)

article_paths = sorted(glob.glob(article_dir + "*"))
summary_paths = sorted(glob.glob(summary_dir + "*"))


for data_type in data_types:
    bbcids = split_dict[data_type]

    if data_type == 'validation':
        dtype = 'val'
    else:
        dtype = data_type
    for bbcid_idx, bbcid in enumerate(tqdm(bbcids)):
        article_path = os.path.join(article_dir, bbcid + '.document')
        summary_path = os.path.join(summary_dir, bbcid + '.summary')
        if not os.path.exists(article_path):
            continue

        with open(article_path) as f:
            article = f.read()
        article = article.replace('''Share this with
Email
Facebook
Messenger
Messenger
Twitter
Pinterest
WhatsApp
LinkedIn
Copy this link
These are external links and will open in a new window
''', '')
        article_sents = [line.strip() for line in article.strip().split('\n')]
        with open(summary_path) as f:
            lines = f.readlines()
        # if len(lines) != 1:
        #     print lines
        #     raise Exception('%s does not have 1 line' % summary_path)
        summary = lines
        out_str = '\n'.join(article_sents) + '\n' + '<SUMMARIES>\n' + '\n'.join(summary)
        out_path = os.path.join(out_dir, '%s_%06d.txt' % (dtype, bbcid_idx))
        with open(out_path, 'wb') as f:
            f.write(out_str)