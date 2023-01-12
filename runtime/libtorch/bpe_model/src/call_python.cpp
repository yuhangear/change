#include <iostream>
#include <Python.h>

int main(int argc, char** argv){
    // 运行Python解释器
    Py_Initialize();
    
    // 添加.py的路径
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/datadrive/wenet/wenet-hotwords/temp/wenet/wenet/runtime/libtorch/bpe_model/')");
	
    /**************************
    ********* add函数 **********
    **************************/
    
    // 导入模块
    PyObject* pModule = PyImport_ImportModule("calculator");
    // 导入要运行的函数
    PyObject* pFunc = PyObject_GetAttrString(pModule, "add");
    // 构造传入参数    
    PyObject* args = PyTuple_New(2);
    PyTuple_SetItem(args, 0, Py_BuildValue("i", 1));
    PyTuple_SetItem(args, 1, Py_BuildValue("i", 10));
    // 运行函数，并获取返回值
    PyObject* pRet = PyObject_CallObject(pFunc, args);
    if (pRet)
    {
        long result = PyLong_AsLong(pRet); // 将返回值转换成long型  
        std::cout << "result:" << result << std::endl ;
    }
	
    /**************************
    ****** helloworld函数 *****
    **************************/
    
	// 导入函数
    pFunc = PyObject_GetAttrString(pModule, "helloworld");
    // 构造传入参数
    PyObject* str = Py_BuildValue("(s)", "python");
    // 执行函数
    PyObject_CallObject(pFunc, str);    
    // 终止Python解释器
    Py_Finalize();  

}

