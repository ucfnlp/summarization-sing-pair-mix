#!/usr/bin/env bash

set -x

intexit() {
    # Kill all subprocesses (all processes in the current process group)
    kill -HUP -$$
}

hupexit() {
    # HUP'd (probably by intexit)
    echo
    echo "Interrupted"
    exit
}

trap hupexit HUP
trap intexit INT


dataset_name=cnn_dm
mode=train
singles_and_pairs=singles

sentemb=True
artemb=True
plushidden=True
batch_size=8

cuda=0
exp_suffix=""
dataset_split=train
num_iterations=500000
port=6006

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset_name=*)
      dataset_name="${1#*=}"
      ;;
    --mode=*)
      mode="${1#*=}"
      ;;
    --singles_and_pairs=*)
      singles_and_pairs="${1#*=}"
      ;;
    --cuda=*)
      cuda="${1#*=}"
      ;;
    --batch_size=*)
      batch_size="${1#*=}"
      ;;
    --sentemb=*)
      sentemb="${1#*=}"
      ;;
    --artemb=*)
      artemb="${1#*=}"
      ;;
    --plushidden=*)
      plushidden="${1#*=}"
      ;;
    *)
        break
  esac
  shift
done

if [[ "$mode" = "all" ]]; then
    mode=train_predict_summ_tensorboard_pg
fi

if [[ "$mode" = "eval" ]]; then
    dataset_split=val
    num_iterations=-1
fi


if [[ "$sentemb" = "True" ]]; then
    exp_suffix="$exp_suffix"_sentemb
fi

if [[ "$artemb" = "True" ]]; then
    exp_suffix="$exp_suffix"_artemb
fi

if [[ "$plushidden" = "True" ]]; then
    exp_suffix="$exp_suffix"_plushidden
fi



if [[ "$cuda" = "1" ]]; then
    port=7007
fi

echo "$dataset_name"
echo "$mode"
echo "$singles_and_pairs"
echo "$artemb"
echo "$@"

if [[ "$mode" == *"tensorboard"* ]]; then
    CUDA_VISIBLE_DEVICES="$cuda" tensorboard --logdir=$HOME/discourse/data/bert/"$dataset_name"/"$singles_and_pairs"/output"$exp_suffix" --port="$port" &
fi
if [[ "$mode" == *"train"* ]]; then
    cd bert
    CUDA_VISIBLE_DEVICES="$cuda" python run_classifier.py   --task_name=merge   --do_train=true   --do_eval=true   --dataset_name="$dataset_name" --singles_and_pairs="$singles_and_pairs"   --max_seq_length=64   --train_batch_size=32   --learning_rate=2e-5   --num_train_epochs=1000.0 --batch_size="$batch_size" --sentemb="$sentemb" --artemb="$artemb" --plushidden="$plushidden"  "$@"
    cd ..
fi
if [[ "$mode" == *"predict"* ]]; then
    cd bert
    CUDA_VISIBLE_DEVICES="$cuda" python run_classifier.py   --task_name=merge   --do_predict=true   --dataset_name="$dataset_name" --singles_and_pairs="$singles_and_pairs"  --max_seq_length=64 --sentemb="$sentemb" --artemb="$artemb"  --plushidden="$plushidden" "$@"
    cd ..
fi
if [[ "$mode" == *"summ"* ]]; then
    python bert_scores_to_summaries.py --dataset_name="$dataset_name" --singles_and_pairs="$singles_and_pairs" --sentemb="$sentemb" --artemb="$artemb"  --plushidden="$plushidden"
fi
if [[ "$mode" == *"pg"* ]]; then
    CUDA_VISIBLE_DEVICES="$cuda" python ssi_to_pg_input.py --dataset_name="$dataset_name" --singles_and_pairs="$singles_and_pairs" --use_bert=True --sentemb="$sentemb" --artemb="$artemb"  --plushidden="$plushidden" "$@"
fi
