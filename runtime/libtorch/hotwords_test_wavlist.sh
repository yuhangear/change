#!/bin/bash

file=/datadrive/temp/ntu_wenet/test_malay_eng_code-switch_cut/
# file=/datadrive/wenet/wenet-hotwords/examples/release/s0/data/imda_part2
chunk_size=-1
sample_rate=16000
is_hotword=true
hotlist=1
model_dir=./model/release_malay-eng-model_500imda_200King_31mix_splice100

decoding_mode="attention_rescoring" # ctc_prefix_beam_search

if [ "$decoding_mode" == "attention_rescoring" ]; then
    rescoring_weight=1.0
else
    rescoring_weight=0.0
fi


export GLOG_logtostderr=1
export GLOG_v=2

context_path=./hot_word_dir/


:>$file/log.txt


if [ $is_hotword == "true" ];
then
	echo "hotword"
	./build/bin/decoder_main \
    	    --wav_scp $file/wav.scp2 \
    	    --chunk_size $chunk_size \
    	    --context_path $context_path \
	    --context_score 3 \
	    --hotlist $hotlist \
    	    --model_path $model_dir/final.zip \
	    --rescoring_weight $rescoring_weight \
    	    --unit_path $model_dir/words.txt2 2>&1  >  $file/temp
else
    
    # --context_path $context_path \
    # --context_score 4 \
    # --context_path $context_path \
    # --context_score 3 \
			# --hotlist 0 \
			#--wav_scp $file/wav.scp \
# --wav_path /datadrive/wenet/wenet-hotwords/examples/release/s0/data/imda_part2/raw_data/026710436.WAV \
    ./build/bin/decoder_main \
    	--wav_scp $file/wav.scp2 \
    	--chunk_size $chunk_size \
    	--model_path $model_dir/final.zip \
	--rescoring_weight $rescoring_weight \ 
    	--unit_path $model_dir/words.txt2 2>&1  >  $file/temp
    
fi

wait
