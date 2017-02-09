// eye_dll_test.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include "Includes.h"
#include <windows.h>
#include <fstream>

#include "../OCR_Api.h"
#pragma comment(lib, "../bin32/ocr_dll.lib") 

using namespace cv;
using namespace std;



int _tmain(int argc, _TCHAR* argv[])
{
    const void *handle = NULL;
    
    int r = -1;
    
    r = OcrInit(&handle, "SVM_type2_calc0.xml", "");

    if (r!=0)
    {
        cout << "init error" << endl;
        return 0;
    }

    for (int i = 0; i < 10; i++)
    {
        Mat img = imread("047255.jpg", CV_LOAD_IMAGE_UNCHANGED);
        ImageData imgData;
        imgData.Channels = img.channels();
        imgData.Buf = img.data;
        imgData.Height = img.rows;
        imgData.Width = img.cols;

        RetStr retStr;
        int ocr_type = 2000;
        //cout << "这是win32 C++对dll的测试，输入ocr_type=3040,dll会返回 12AB 的字符串。" << endl;

        cout << "这是win32 C++对dll的测试，输入ocr_type=2000,dll会返回结果" << endl;
        int r = OcrRecognition(handle, &imgData, &retStr, ocr_type);
        //for (int i = 0; i < retStr.RetCnt; i++)
        //{
        //    cout << retStr.RetBuf[i] << " ";
        //}
        int ret = (int)retStr.RetBuf[0];
        cout << "结果是" << ret << endl;
    }
    

	return 0;
}
