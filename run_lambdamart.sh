#!/usr/bin/env bash


#intexit() {
#    # Kill all subprocesses (all processes in the current process group)
#    kill -HUP -$$
#}
#
#hupexit() {
#    # HUP'd (probably by intexit)
#    echo
#    echo "Interrupted"
#    exit
#}

#trap hupexit HUP
#trap intexit INT


SINGLES_AND_PAIRS=singles
DATASET_NAME=cnn_dm

while [ $# -gt 0 ]; do
  case "$1" in
    --SINGLES_AND_PAIRS=*)
      SINGLES_AND_PAIRS="${1#*=}"
      ;;
    --DATASET_NAME=*)
      DATASET_NAME="${1#*=}"
      ;;
    *)
        break
  esac
  shift
done

echo "$SINGLES_AND_PAIRS"
echo "$DATASET_NAME"

if [ "$DATASET_NAME" == 'duc_2004' ]
then
    MODEL_DATASET_NAME=cnn_dm
    THREADS="lambdamart_scores_to_summaries.py"
else
    MODEL_DATASET_NAME="$DATASET_NAME"
    THREADS="-m scoop -vv --hostfile hosts_32_threads lambdamart_scores_to_summaries.py"
fi

echo "$MODEL_DATASET_NAME"
echo "$THREADS"

set -x

java -Xmx120g -jar ~/ranklib/bin/RankLib-2.10.jar -train data/to_lambdamart/"$DATASET_NAME"_"$SINGLES_AND_PAIRS"/train_10000.txt -ranker 6 -metric2t NDCG@5 -metric2T NDCG@5 -save data/lambdamart_models/"$DATASET_NAME"_ndcg5_"$SINGLES_AND_PAIRS"_10000_tree100.txt -sparse -estop 50 -leaf 10 -tree 50

#java -Xmx120g -jar ~/ranklib/bin/RankLib-2.10.jar -rank data/temp/"$DATASET_NAME"/to_lambdamart/lambdamart_"$SINGLES_AND_PAIRS".txt -score data/temp/"$DATASET_NAME"/lambdamart_results/lambdamart_"$SINGLES_AND_PAIRS".txt -ranker 6 -metric2t NDCG@5 -metric2T NDCG@5  -load data/lambdamart_models/"$MODEL_DATASET_NAME"_ndcg5_"$SINGLES_AND_PAIRS"_10000.txt -sparse && python lambdamart_scores_to_summaries.py --dataset_name="$DATASET_NAME" --singles_and_pairs="$SINGLES_AND_PAIRS" --mode=generate_summaries

#python -m scoop -vv --hostfile hosts_32_threads preprocess_for_lambdamart_no_flags.py --singles_and_pairs="$SINGLES_AND_PAIRS" --dataset_name="$DATASET_NAME" && head -10000 data/to_lambdamart/"$DATASET_NAME"_"$SINGLES_AND_PAIRS"/train.txt > data/to_lambdamart/"$DATASET_NAME"_"$SINGLES_AND_PAIRS"/train_10000.txt && java -Xmx120g -jar ~/ranklib/bin/RankLib-2.10.jar -train data/to_lambdamart/"$DATASET_NAME"_"$SINGLES_AND_PAIRS"/train_10000.txt -validate data/to_lambdamart/"$DATASET_NAME"_"$SINGLES_AND_PAIRS"/val.txt -ranker 6 -metric2t NDCG@5 -metric2T NDCG@5 -save data/lambdamart_models/"$DATASET_NAME"_ndcg5_"$SINGLES_AND_PAIRS"_10000.txt -sparse -estop 200 && python "$THREADS" lambdamart_scores_to_summaries.py --dataset_name="$DATASET_NAME" --singles_and_pairs="$SINGLES_AND_PAIRS" --mode=write_to_file && java -Xmx120g -jar ~/ranklib/bin/RankLib-2.10.jar -rank data/temp/"$DATASET_NAME"/to_lambdamart/lambdamart_"$SINGLES_AND_PAIRS".txt -score data/temp/"$DATASET_NAME"/lambdamart_results/lambdamart_"$SINGLES_AND_PAIRS".txt -ranker 6 -metric2t NDCG@5 -metric2T NDCG@5  -load data/lambdamart_models/"$DATASET_NAME"_ndcg5_"$SINGLES_AND_PAIRS"_10000.txt -sparse && python lambdamart_scores_to_summaries.py --dataset_name="$DATASET_NAME" --singles_and_pairs="$SINGLES_AND_PAIRS" --mode=generate_summaries