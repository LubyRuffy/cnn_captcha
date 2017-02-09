// eye_dll_test.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include "Includes.h"
#include <windows.h>
#include <fstream>


using namespace cv;
using namespace std;


#define LITTLE_ENDIAN (1)
#define BIG_ENDIAN (0)
/*return 1: little-endian, return 0: big-endian*/
int checkCPUendian()
{
    union
    {
        unsigned int a;
        unsigned char b;
    }c;
    c.a = 1;
    return (c.b == 1);
}

int _tmain(int argc, _TCHAR* argv[])
{
    cout << checkCPUendian() << endl;
    
	return 0;
}
