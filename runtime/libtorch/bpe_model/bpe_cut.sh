for i in hot_word_dir_old/* ; do 
out=`echo $i | cut -d "/" -f2 `
python bpe_model/bpe_cut2.py hot_word_dir_old/$out hot_word_dir/$out 

done
