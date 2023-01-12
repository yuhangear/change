#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=0 # start from 0 if you need to start from data preparation
stop_stage=0

nj=16

model_dir=/home3/yuhang001/new_wenet/wenet/examples/gigaspeech/release_singapo_eng/sp_spec_aug_train-oct-18_big_model/
dict=$model_dir/units.txt
checkpoint=$model_dir/avg_10.pt
config=$model_dir/train.yaml
bpe_model=/home/asrxiv/w2022/projects/wenet-gigaspeech-16k/data/lang_char_XL/train_xl_unigram5000.model

dir=exp/
wave_data=raw_wav
set=imda-part2
ali_format=$wave_data/$set/data.list_temp
ali_result=$dir/ali
hot_weight=3
decoding_type="attention_rescoring_word_boundary"
. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    nj=32
    # Prepare required data for ctc alignment
    echo "Prepare data, prepare required format"
    for x in $set; do
    tools/make_raw_list.py $wave_data/$x/wav.scp $wave_data/$x/text --segments $wave_data/$x/segments \
        $wave_data/$x/data.list

    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then

    # Test model, please specify the model you want to use by --checkpoint
        python wenet/bin/word_alignment.py --gpu -1 \
            --decoding_type $decoding_type \
            --config $config \
            --bpe_model $bpe_model \
            --gen_praat \
            --input_file $ali_format \
            --checkpoint $checkpoint \
            --batch_size 1 \
            --dict $dict \
            --hot_weight $hot_weight \
            --result_file $ali_result 

fi