/*******************************************************************************/
/*  Copyright: (C) 2015 北京七鑫易维信息技术有限公司
/*
/*  File: GlobalDefine.h
/*
/*  Author: LiuHui
/*
/*  Date: 2016-06-12
/*
/*  Description: 
/*******************************************************************************/

#ifndef __INCLUDES_H__
#define __INCLUDES_H__


//#define ANDROID_STATIC_IMAGE_DEBUG 1 //

//#define PC_WIN7_DEBUG
//#define MOBILE_ANDROID_DEBUG 1

//#define USING_DATA_ANALYSIS
//#define ZTE_PHONE

//#define MEMTEST


#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif


#ifndef LONG
typedef long LONG;
#endif

#ifndef BYTE
typedef unsigned char BYTE;
#endif

#ifndef LPVOID
typedef void* LPVOID;
#endif


#define  M_PI  3.14159265358979323846


#include <mutex>

#include <new>
#include <map>
#include <list>
#include <queue>
#include <limits>
#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
using namespace std;

#include<opencv2/opencv.hpp>
using namespace cv;


#endif//__INCLUDES_H__
