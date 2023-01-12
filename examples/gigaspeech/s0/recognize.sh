#!/bin/bash

# Copyright 2021 Mobvoi Inc. All Rights Reserved.

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

model_dir=/home3/yuhang001/new_wenet/wenet/examples/gigaspeech/release_singapo_eng/wenet_Sgeng_16khz_12-Nov-2022_train-nov-12_v1/
dict=$model_dir/units.txt
checkpoint=$model_dir/avg_model.pt
decode_checkpoint=$checkpoint
config=$model_dir/train.yaml
bpemodel=/home/asrxiv/w2022/projects/wenet-gigaspeech-16k/data/lang_char_XL/train_xl_unigram5000
wav_name=imda-part2_small

dir=exp/
wave_data=raw_wav
set=$wav_name
ali_format=$wave_data/$set/data.list
ali_result=$dir/ali
recog_set=$wav_name
data=raw_wav/
decode_modes="ctc_prefix_beam_search_hot"
hot_weight=3
. tools/parse_options.sh || exit 1;


  # Test model, please specify the model you want to test by --checkpoint
  cmvn_opts=
  $cmvn && cmvn_opts="--cmvn data/${train_set}/global_cmvn"
  # TODO, Add model average here
  mkdir -p $dir/test

  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=-1
  ctc_weight=0.5
  # Polling GPU id begin with index 0
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  idx=0
  for test in $recog_set; do
    for mode in ${decode_modes}; do
    {
      {
        test_dir=$dir/${test}_${mode}_${hot_weight}
        mkdir -p $test_dir
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
        python wenet/bin/recognize.py --gpu $gpu_id \
          --mode $mode \
          --config $config \
          --data_type "raw" \
          --bpe_model $bpemodel.model \
          --test_data $ali_format \
          --checkpoint $decode_checkpoint \
          --beam_size 20 \
          --batch_size 1 \
          --penalty 0.0 \
          --dict $dict \
          --result_file $test_dir/text_bpe \
          --ctc_weight $ctc_weight \
          --hot_weight $hot_weight \
          ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}



        cut -f2- -d " " $test_dir/text_bpe > $test_dir/text_bpe_value_tmp
        cut -f1 -d " " $test_dir/text_bpe > $test_dir/text_bpe_key_tmp

        tools/spm_decode --model=${bpemodel}.model --input_format=piece \
          < $test_dir/text_bpe_value_tmp | sed -e "s/▁/ /g" > $test_dir/text_value
        paste -d " " $test_dir/text_bpe_key_tmp $test_dir/text_value > $test_dir/text
        # a raw version wer without refining processs
        python tools/compute-wer.py --char=1 --v=1 \
          $data/$test/text $test_dir/text > $test_dir/wer


      } 

    }
    done
  done

