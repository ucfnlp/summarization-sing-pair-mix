import os


with open(os.path.expanduser('~') + "/discourse/data/coref/xsum/corenlp_lists/all_train.txt") as f:
  urls = f.readlines()
necessary_urls = []
for url in urls:
  # Get file id
  necessary_urls.append(url.strip().split("/")[-1].split(".")[0])
print(len(necessary_urls), necessary_urls[0])

file_names = [filename for filename in os.listdir(os.path.expanduser('~') + "/discourse/data/coref/xsum/processed") if 'train' in filename]
collected_fileids = [file_name.split(".")[0] for file_name in file_names]

print(len(collected_fileids), collected_fileids[0])

missing_files = list(set(necessary_urls) - set(collected_fileids))
missing_file_names = ["data/coref/xsum/to_coref/" + file_name + ".bin" for file_name in missing_files]


with open(os.path.expanduser('~') + "/discourse/data/coref/xsum/corenlp_lists/missing_list.txt", "wb") as f:
  f.write("\n".join(missing_file_names))