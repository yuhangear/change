#!/bin/bash

export GLOG_logtostderr=1
export GLOG_v=2
decoding_mode="attention_rescoring" # ctc_prefix_beam_search

if [ "$decoding_mode" == "attention_rescoring" ]; then
    rescoring_weight=1.0
else
    rescoring_weight=0.0
fi

model_dir=model/release_malay-eng-model_500imda_200King_31mix_splice100
context_path=./hot_word_dir/
./build/bin/websocket_server_main \
    --port 20089 \
    --bpe_path /datadrive/temp/temp/wenet1/runtime/libtorch/bpe_model    \
    --chunk_size -1    \
    --model_path $model_dir/final.zip    \
    --context_path $context_path   \
    --context_score 3  \
    --rescoring_weight $rescoring_weight \
    --unit_path $model_dir/words.txt2 2>&1 | tee server.log



