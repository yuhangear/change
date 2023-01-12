import sentencepiece as spm
sp = spm.SentencePieceProcessor()

import math


import sys
infile=sys.argv[1]


sp.load("/datadrive/wenet/wenet-hotwords/temp/wenet/wenet/runtime/libtorch/bpe_model/train_unigram5000.model")

# spm.spm_export_vocab() #--output="/datadrive/wenet/wenet-hotwords/temp/wenet/wenet/runtime/libtorch/bpe_model/va"


print(sp.encode_as_pieces(infile))
