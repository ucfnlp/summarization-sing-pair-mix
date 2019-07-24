# Scoring Sentence Singletons and Pairs for Abstractive Summarization
Code for the ACL 2019 paper "Scoring Sentence Singletons and Pairs for Abstractive Summarization"

# Data

Code and models coming soon!

python run_summarization.py --mode=train --dataset_name=cnn_dm --dataset_split=train --exp_name=cnn_dm --singles_and_pairs=both --max_enc_steps=100 --min_dec_steps=10 --max_dec_steps=30 --single_pass=False --batch_size=128 --num_iterations=10000000 --by_instance=True



python bert/extract_features.py --layers=-1,-2,-3,-4 --max_seq_length=400 --batch_size=1

python run_classifier.py --task_name=merge --do_train=true --do_eval=true --dataset_name=cnn_dm --singles_and_pairs=both --max_seq_length=64 --train_batch_size=32 --batch_size=8 --learning_rate=2e-5 --num_train_epochs=1000.0 --sentemb=True

python run_classifier.py --task_name=merge --do_predict=true --dataset_name=cnn_dm --singles_and_pairs=both --max_seq_length=64 --sentemb=True

python bert_scores_to_summaries.py --dataset_name=cnn_dm --singles_and_pairs=both --sentemb=True

python ssi_to_pg_input.py --dataset_name=cnn_dm --singles_and_pairs=both --use_bert=True --sentemb=True