import sentencepiece as spm
sp = spm.SentencePieceProcessor()
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
import math 


import sys
infile=sys.argv[1]
outfile=sys.argv[2]
sp.load("/home/asrxiv/w2022/projects/wenet-gigaspeech-16k/data/lang_char_XL/train_xl_unigram5000.model")

symbol_table = read_symbol_table("/home/asrxiv/w2022/projects/wenet-gigaspeech-16k/data/lang_char_XL/train_xl_unigram5000_units.txt")

outfile=open(outfile,"w")
with open(infile,"r") as f :
    for line in f:
        line=line.strip()
        line=sp.encode_as_pieces(line)
        new_line=""
        for i in line:
            new_line=new_line+" "+str(symbol_table[i])

        outfile.writelines(new_line+"\n")
outfile.close()


