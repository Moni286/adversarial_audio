#!/bin/bash

if [ $# -lt 5 ]
    then
    echo "Usage: $(basename $0) source_label target_label limit max_iters test_size."
    exit 1
fi
dataset_dir="data"
ckpts_dir="ckpts"
source_label=$1
target_label=$2
limit=$3
max_iters=$4
test_size=$5


frozen_graph="$ckpts_dir/conv_actions_frozen.pb"
labels_file="$ckpts_dir/conv_actions_labels.txt"

mkdir -p "output/data/$source_label"
find "$dataset_dir/$source_label/" -name "*.wav" | sort -R \
    | head -n$test_size | xargs -L1 cp -t "output/data/$source_label"

mkdir -p "output/data/$target_label"
find "$dataset_dir/$target_label/" -name "*.wav" | sort -R \
    | head -n$test_size | xargs -L1 cp -t "output/data/$target_label"

echo "Running attack: $source_label --> $target_label"
output_dir="output/result/$target_label/$source_label"
mkdir -p $output_dir
python3 stft_audio_attack.py \
--data_dir="output/data/$source_label" \
--output_dir=$output_dir \
--target_label=$target_label \
--labels_path=$labels_file \
--graph_path=$frozen_graph \
--limit=$limit \
--max_iters=$max_iters
