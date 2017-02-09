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


#if defined(_BUILD_OCR_DLL_) && !defined(APP_TEST_API)
#		define APP_TEST_API __declspec (dllexport) // 实现者应定义该宏（VS中通过预处理器定义）
#	else
#		define APP_TEST_API __declspec (dllimport)
#endif



#include <cstdint>



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
    unsigned char   RetBuf[16];   //返回数组 最多返回10个
    int     RetCnt;
}RetStr, *PRetStr;


#ifdef __cplusplus
extern "C"
{
#endif//__cplusplus


    /*****************************************************************************
    *                       OcrInit
    *  函 数 名： OcrInit
    *  说    明：
    *  参    数：handle:句柄
    *            path1:路径1
    *            path2:路径2
    *  返 回 值：0 ： 成功，非零 ： 失败
    *****************************************************************************/
    APP_TEST_API int32_t __stdcall OcrInit(const void** handle, const char* path1, const char* path2);


    /*****************************************************************************
    *                       OcrRelease
    *  函 数 名： OcrRelease
    *  说    明：
    *  参    数：handle:句柄
    *  返 回 值：0 ： 成功，非零 ： 失败
    *****************************************************************************/
    APP_TEST_API int32_t __stdcall OcrRelease(void* handle);

	
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
    APP_TEST_API int32_t __stdcall OcrRecognition(const void* handle, const ImageData * const PImageData, PRetStr pPRetStr, int ocr_type);

#ifdef __cplusplus
}//extern "C"
#endif//__cplusplus


#endif//__INCLUDE_OCR_API_H__