#include<iostream>
#include <Python.h>
//#include "/usr/include/python3.8/Python.h"

int main()
{
	Py_Initialize();
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("import sentencepiece as spm");
	PyRun_SimpleString("sp = spm.SentencePieceProcessor()");
	PyRun_SimpleString("import math");



	PyRun_SimpleString("sys.path.append('/datadrive/wenet/wenet-hotwords/temp/wenet/wenet/runtime/libtorch/bpe_model')");
	PyRun_SimpleString("import bpe_cut");  // 导入py文件
	PyRun_SimpleString("bpe_cut.cut('/datadrive/wenet/wenet-hotwords/temp/wenet/wenet/runtime/libtorch/hot_word_dir_old/hot1','/datadrive/wenet/wenet-hotwords/temp/wenet/wenet/runtime/libtorch/hot_word_dir/hot4')");  // 调用python函数
	Py_Finalize();
	system("pause");
	return 0;
}
