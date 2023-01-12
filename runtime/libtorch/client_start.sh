#!/bin/bash
hot_list=0
sample_rate=16000
. utils_kaldi/parse_options.sh

export GLOG_logtostderr=1
export GLOG_v=2
# wav_path=/datadrive/wenet/wenet-hotwords/examples/release/s0/data/ntu-conversation/raw_wav/S1-mml-1-dec-2017-a-session1-2-09365-15329.wav

# wav_path=/datadrive/wenet/wenet-hotwords/examples/release/s0/data/imda_part2/raw_data/026710436.WAV
wav_path=/datadrive/wenet/wenet-hotwords/examples/release/s0/data_8k/imda_part2/raw_wav2/nsc-part2-chn0-spk03092658maf-026581716.wav
#SEMBAWANG SHIPYARD LORONG ABU TALIB AND GREAT WORLD APARTMENTS


#hot word  : LORONG ABU TALIB





# ./build/bin/websocket_client_main  --hot_list $hot_list --sample_rate $sample_rate  --hostname 127.0.0.1 --port 20089 --continuous_decoding=true   --wav_path $wav_path 2>&1 | tee client.log

./test.py -u ws://localhost:$port $wav_path

                                                                                                                                                                                                                                                                                                                                              
