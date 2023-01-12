#!/bin/bash

# file=/datadrive/wenet/wenet-hotwords/temp/wenet/wenet/runtime/libtorch/data/imda_part2/
file=
chunk_size=32
sample_rate=16000
is_hotword=true
hotlist=1
model_dir=
eos_symbol_id=`grep '<sos/eos>' $model_dir/units.txt | cut -d " " -f2`
echo $eos_symbol_id
export GLOG_logtostderr=1
export GLOG_v=2

decoding_mode="attention_rescoring" # ctc_prefix_beam_search

if [ "$decoding_mode" == "attention_rescoring" ]; then
    rescoring_weight=1.0
else
    rescoring_weight=0.0
fi

context_path=./hot_word_dir/

# model_dir=/path/to/model
:>$file/log.txt


if [ $is_hotword == "true" ];
then
	echo "hotword"
	./build/bin/decoder_main \
    	    --wav_scp $file/wav.scp \
    	    --chunk_size $chunk_size \
    	    --thread_num 1 \
	    --eos_symbol_id "$eos_symbol_id" \
	    --context_path $context_path \
	    --context_score 3 \
	    --hotlist 1 \
	    --rescoring_weight $rescoring_weight \
	    --model_path $model_dir/final.zip \
	    --unit_path $model_dir/words.txt 2>&1  >  $file/temp
else
    #--eos_symbol_id 4998 \
    # --context_path $context_path \
    # --context_score 4 \
    # --context_path $context_path \
    # --context_score 3 \
  	./build/bin/decoder_main \
    	    --wav_scp $file/wav.scp \
    	    --chunk_size $chunk_size \
	    --eos_symbol_id $eos_symbol_id \
    	    --model_path $model_dir/final.zip \
	    --thread_num 1 \
	    --rescoring_weight $rescoring_weight \
    	    --unit_path $model_dir/words.txt 2>&1  >  $file/temp

fi

wait
