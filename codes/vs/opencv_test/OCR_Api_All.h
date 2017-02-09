/******************************************************************************************
** 文件名:  OCR_Api.h
×× 主要类:    
**
** Copyright (c) 
** 创建人:
** 日  期:
** 修改人:
** 日  期:
** 描  述:   OCR_Api
*××  
**
** 版  本:   1.0.0
** 备  注:
**
*****************************************************************************************/

#ifndef __INCLUDE_OCR_API_H__
#define __INCLUDE_OCR_API_H__


#if defined(_MSC_VER) || defined(WIN32)
#	if defined(_BUILD_OCR_DLL_) && !defined(APP_TEST_API)
#		define APP_TEST_API __declspec (dllexport) // 实现者应定义该宏（VS中通过预处理器定义）
#		define CALL_OCR __stdcall
#	elif defined(_BUILD_STATIC_OCR_LIB_) && !defined(APP_TEST_API)
#		define APP_TEST_API  // 实现者应定义该宏（VS中通过预处理器定义）
#		define CALL_OCR 
#	else
#		define APP_TEST_API __declspec (dllimport)
#		define CALL_OCR __stdcall
#	endif
#else
#	define CALL_OCR
#	define APP_TEST_API // 非windows编译环境
#endif


#include <cstdint>



enum OCR_ERROR
{
	OCR_OK = 0,
    OCR_ERROR_OVERFLOW = 1,
	OCR_ERROR_PARAMER = 2,
	OCR_ERROR_CHANNELS = 3,
    OCR_ERROR_TYPE = 4,
};
#define OCR_RET(A) (int32_t)(A)


//图像传输控制结构体
typedef struct __ImageData
{
    unsigned char * Buf;  //图像数据流
	short Width;          //图像的宽度
	short Height;         //图像的高度
	int   Channels;       //图像通道数
}ImageData, *PImageData;


//图像传输控制结构体
typedef struct __RetStr
{
    unsigned char RetBuf[16];   //返回数组 最多返回10个
    int           RetCnt;
}RetStr, *PRetStr;


#ifdef __cplusplus
extern "C"
{
#endif//__cplusplus


	/*****************************************************************************
    *                       OCR
    *  函 数 名： OcrRecognition
    *  说    明：
    *  参    数：PImageData:图像数据
    *            pPRetStr:输出ocr=结果
    *            ocr_type:ocr类型
    *                     英文数字混合 ocr_type=3000 任意长度英数混合，识别率会降低
    *                     ocr_type=3010 1位英数
    *                     ocr_type=3020 2位英数混合
    *                     ocr_type=3100 10位英数混合
	*  返 回 值：0 ： 成功，非零 ： 失败
	*****************************************************************************/
    APP_TEST_API int32_t __stdcall OcrRecognition(const ImageData * const PImageData, PRetStr pPRetStr, int ocr_type);

#ifdef __cplusplus
}//extern "C"
#endif//__cplusplus


#endif//__INCLUDE_OCR_API_H__