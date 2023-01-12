#!/bin/bash

file=
chunk_size=16
sample_rate=16000
is_hotword=false
. utils/parse_options.sh


export GLOG_logtostderr=1
export GLOG_v=2

context_path=/datadrive/wenet/wenet-hotwords/examples/release/s0/content_text


if [ $sample_rate == "8000" ];
then
	model_dir=/datadrive/wenet/wenet-hotwords/examples/release/s0/exp_8k/sp_spec_aug
	:>$file/log.txt
	cd /datadrive/wenet/wenet-hotwords/runtime_8k/server/x86
else
	model_dir=/datadrive/wenet/wenet-hotwords/examples/release/s0/exp_16k/sp_spec_aug
	:>$file/log.txt
	cd /datadrive/wenet/wenet-hotwords/runtime/server/x86
fi


if [ $is_hotword == "true" ];
then
	echo "hotword"
	./build/decoder_main \
    	--wav_scp $file/wav.scp \
    	--chunk_size $chunk_size \
    	--context_path $context_path \
		--context_score 3 \
    	--model_path $model_dir/final.zip \
    	--dict_path $model_dir/words.txt 2>&1  >  $file/temp
else
    # --context_path $context_path \
    # --context_score 4 \
    # --context_path $context_path \
    # --context_score 3 \
  	./build/decoder_main \
    	--wav_scp $file/wav.scp \
    	--chunk_size $chunk_size \
    	--model_path $model_dir/final.zip \
    	--dict_path $model_dir/words.txt 2>&1  >  $file/temp

fi

wait




