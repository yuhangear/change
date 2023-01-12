import sentencepiece as spm
sp = spm.SentencePieceProcessor()

import math


import sys
infile=sys.argv[1]
outfile=sys.argv[2]



sp.load("/datadrive/wenet/wenet-hotwords/temp/wenet/wenet/runtime/libtorch/bpe_model/train_unigram5000.model")
outfile=open(outfile,"w")
with open(infile,"r") as f :
    for line in f:
        line=line.strip()
        line=sp.encode_as_pieces(line)
        new_line=""
        for i in line:
            new_line=new_line+" "+i



        outfile.writelines(new_line+"\n")
outfile.close()
