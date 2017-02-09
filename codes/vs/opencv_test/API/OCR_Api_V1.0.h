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

#ifdef __cplusplus
extern "C"
{
#endif//__cplusplus


	/*****************************************************************************
	*                             初始化
	*  函 数 名：OcrInit
	*  功    能：人眼检测算法初始化
	*  说    明：
	*  参    数：path1:在adaboost算法中，path1为xml文件路径
    *                  在darknet算法中，path1为cfg文件路径
	*            path2:在adaboost算法中，path2为model文件路径
    *                  在darknet算法中，path2为weight文件路径
	*            pDevParam:设备相关变量指针
    *            pStatusParam:状态相关变量指针
	*  返 回 值：0 ： 成功，非零 ： 失败
	*  创 建 人：liuh
	*  创建时间：2016-08-24
	*  修 改 人：
	*  修改时间：
	*****************************************************************************/
    APP_TEST_API int32_t CALL_OCR OcrInit(const char* path1, const char* path2, void*& pDevParam, void*& pStatusParam);


	/*****************************************************************************
	*                       释放资源
	*  函 数 名：OcrRelease
	*  功    能：释放眼睛检测库使用的相关资源
	*  说    明：
	*  参    数：pDevParam:设备变量指针
    *            pStatusParam:状态变量指针
	*  返 回 值： 0 ： 成功，非零 ： 失败
	*  创 建 人：liuh
	*  创建时间：2016-08-24
	*  修 改 人：
	*  修改时间：
	*****************************************************************************/
    APP_TEST_API int32_t CALL_OCR OcrRelease(const void* pDevParam, const void* pStatusParam);


	/*****************************************************************************
	*                       眼睛检测
	*  函 数 名：OcrDetection
	*  功    能：实现眼睛检测的库接口
	*  说    明：
    *  参    数：pUsedImage:图像数据结构体指针
    *            pDetOcrs:输出人眼检测算法结果
    *            pDevParam:设备参数指针
    *            pStatusParam:状态参数指针
	*  返 回 值：0 ： 成功，非零 ： 失败
	*  创 建 人：liuh
	*  创建时间：2016-08-24
	*  修 改 人：
	*  修改时间：
	*****************************************************************************/
	APP_TEST_API int32_t CALL_OCR OcrDetection(const UsedImageData * const pUsedImage, PDetOcrs pDetOcrs, const void* pDevParam, const void* pStatusParam);

#ifdef __cplusplus
}//extern "C"
#endif//__cplusplus


#endif//__INCLUDE_OCR_API_H__