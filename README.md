# Scoring Sentence Singletons and Pairs for Abstractive Summarization
Code, Models, and Data for the ACL 2019 paper **"[Scoring Sentence Singletons and Pairs for Abstractive Summarization](https://www.aclweb.org/anthology/P19-1209)"**

# Data

Our data consists of > 1 million **sentence fusion** instances, of the form: 

    Input: one or two articles sentences

    Output: the summary sentence formed by compressing/fusing the input sentences

Our data is derived from existing summarization datasets: CNN/Daily Mail, XSum, and DUC-04.

# Generating the data

We provide instructions on generating the data for CNN/DailyMail. Email the authors at loganlebanoff@knights.ucf.edu if you are interested in our data for XSum and DUC-04.

To generate the CNN/DailyMail data, follow the following steps:

1) Download the zip files from https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail. Unzip and place `cnn_stories_tokenized/` and `dm_stories_tokenized/` inside the `./data/cnn_dm_unprocessed/` directory.
2) Run the following command to extract the articles and summaries from the CNN/DailyMail data. This will create the `articles.tsv` and `summaries.tsv` files.

    ```
    python create_cnndm_dataset.py
    ```
3) Run the following command to run our heuristic, which decides which article sentences were used to create each of the summary sentences. This will create the `highlights.tsv` and `pretty_html.html` files.

    ```
    python highlight_heuristic.py --dataset_name=cnn_dm
    ```

Four files will appear in the newly-created ./data/cnn_dm_processed directory. Each of the .tsv files will have the same number of lines.

* `articles.tsv`
    * 1 article per line, tab-separated list of sentences. Each sentence is tokenized, where each token is separated by a space (" ").

* `summaries.tsv`
    * 1 summary per line, tab-separated list of sentences. Each sentence is tokenized, where each token is separated by a space (" ").

* `highlights.tsv`
    * 1 set of fusion examples per line, tab-separated list of fusion examples. The number of fusion examples corresponds to the same number of summary sentences in the same line in summaries.tsv. Each fusion example is either 0, 1, or 2 source indices (separated by ",") representing which sentences from the article were fused together to form the summary sentence.

* `pretty_html.html`
    * Examples highlighting which sentences were fused together to form summary sentences

# I have my own data, and I want to generate the singletons/pairs of sentences from my data

Follow the steps in "Generating the data," but you may skip steps 1 and 2. Instead, you must generate the two files "article.tsv" and "summaries.tsv" (format is described above after step 3). After you have created those two files, then run step 3.


# Summary Outputs and Models

Summary outputs and models: https://www.dropbox.com/sh/g34aj101oauwlx3/AAA9dNhGlgVDQFiQrAqfMaKIa/Scoring%20Sentence%20Singletons%20and%20Pairs%20for%20Abstractive%20Summarization?dl=0

It includes the following (only for CNN/DailyMail):

- BERT-Extr summaries (see paper for more details)

- BERT-Abs w/ PG summaries (see paper for more details)

- Content Selection model (corresponds to BERT-Extr)

- Sentence Fusion model
(The sentence fusion model reads from a file `logs/cnn_dm_bert_both/ssi.pkl` which contains the sentence singletons/pairs that were chosen by the content selection model.)

# How to run the code

## Data preprocessing

1) Convert the line-by-line data from the *Data* section to TF examples, which are used by the rest of the code.

    ```
    python convert_data.py --dataset_name=cnn_dm
    ```

2) Additionally, convert data to a .tsv format to be input to BERT.

    ```
    python preprocess_for_bert_fine_tuning.py --dataset_name=cnn_dm
    python preprocess_for_bert_article.py --dataset_name=cnn_dm
    python bert/extract_features.py --dataset_name=cnn_dm --layers=-1,-2,-3,-4 --max_seq_length=400 --batch_size=1
    ```

## Training

### Training content selection model

1) Download the pre-trained BERT-Base Uncased model from (https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip), unzip, and place in the `pretrained_bert_model/` directory.

2) Train the BERT model to predict whether a given sentence singleton/pair is likely to create a summary sentence or not.

    ```
    cd bert/
    python run_classifier.py --task_name=merge --do_train --do_eval --dataset_name=cnn_dm --num_train_epochs=1000.0
    cd ..
    ```
    

### Training sentence fusion model

1) Following a similar training procedure to See et al. (https://github.com/abisee/pointer-generator#run-training), train the Pointer-Generator model on sentence singletons/pairs.
    ```
    python run_summarization.py --mode=train --dataset_name=cnn_dm --dataset_split=train --exp_name=cnn_dm --max_enc_steps=100 --min_dec_steps=10 --max_dec_steps=30 --single_pass=False --batch_size=128 --num_iterations=10000000 --by_instance
    ```
    You can perform concurrent evaluation alongside the training process.

    ```
    python run_summarization.py --mode=eval --dataset_name=cnn_dm --dataset_split=val --exp_name=cnn_dm --max_enc_steps=100 --min_dec_steps=10 --max_dec_steps=30 --single_pass=False --batch_size=128 --num_iterations=10000000 --by_instance
    ```
    

2) Once the evaluation loss begins to increase, stop training and evaluation. Then run the following command to restore the model with the lowest evaluation loss.
    ```
    python run_summarization.py --mode=train --dataset_name=cnn_dm --dataset_split=train --exp_name=cnn_dm --max_enc_steps=100 --min_dec_steps=10 --max_dec_steps=30 --single_pass=False --batch_size=128 --num_iterations=10000000 --by_instance --restore_best_model
    ```

## Prediction/Evaluation

### Content selection prediction

1) Run BERT on test examples. This gives a score for every possible sentence in the article and every possible pair of sentences.
    ```
    cd bert/
    python run_classifier.py --task_name=merge --do_predict=true --dataset_name=cnn_dm
    cd ..
    ```

2) Consolidate the scores generated from BERT to create an extractive summary. This also generates a file `ssi.pkl` which contains the sentence singletons/pairs that were selected by BERT.
    ```
    python bert_scores_to_summaries.py --dataset_name=cnn_dm
    ```

### Sentence fusion prediction

1) Run the Pointer-Generator model on the sentence singletons/pairs chosen by BERT in the previous step. The PG model will compress sentence singletons and fuse together sentence pairs.
    ```
    python sentence_fusion.py --dataset_name=cnn_dm --use_bert=True
    ```



