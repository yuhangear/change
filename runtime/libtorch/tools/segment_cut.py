#!/usr/bin/env python3

# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--segments', default=None, help='segments file')
    parser.add_argument('wav_file', help='wav file')
    parser.add_argument('text_file', help='text file')
    parser.add_argument('output_file', help='output list file')
    args = parser.parse_args()
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--segments', default="data/msf-dhl-1hr-testset/segments", help='segments file')
    # parser.add_argument('--wav_file', default="data/msf-dhl-1hr-testset/wav.scp",help='wav file')
    # parser.add_argument('--text_file', default="data/msf-dhl-1hr-testset/text",help='text file')
    # parser.add_argument('--output_file', default="data/msf-dhl-1hr-testset/segment_cut.sh",help='output list file')
    # args = parser.parse_args()

    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            wav_table[arr[0]] = arr[1]

    if args.segments is not None:
        segments_table = {}
        with open(args.segments, 'r', encoding='utf8') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 4
                segments_table[arr[0]] = (arr[1], float(arr[2]), float(arr[3]))
    target_dir=args.wav_file.split("/")[0]+"/"+args.wav_file.split("/")[1]+"/"
    with open(args.text_file, 'r', encoding='utf8') as fin, \
         open(args.output_file, 'w', encoding='utf8') as fout:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]
            txt = arr[1] if len(arr) > 1 else ''

            assert key in segments_table
            wav_key, start, end = segments_table[key]
            wav = wav_table[wav_key]
            #line = dict(key=key, wav=wav, txt=txt, start=start, end=end)
            json_line = "sox "+ str(wav)+" "+ target_dir+"segment_dir/"+str(key)+".wav trim " +  str(start)+" " + str(end-start)
            fout.write(json_line + '\n')
