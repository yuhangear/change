#!/bin/bash

file=/datadrive/temp/ntu_wenet/test_malay_eng_code-switch_cut/
# file=/datadrive/wenet/wenet-hotwords/examples/release/s0/data/imda_part2

# ./hotwords_test_wavlist.sh --file $file

wait

cp $file/temp ./result

python tools/compute-wer.py --char=1 --v=1     $file/text ./result  > ./wer


