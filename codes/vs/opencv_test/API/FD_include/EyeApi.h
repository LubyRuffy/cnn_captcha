/******************************************************************************************
** 文件名:  EyeApi.h
×× 主要类:    
**
** Copyright (c) 七鑫易维有限公司
** 创建人:liuh
** 日  期:2016-08-24
** 修改人:
** 日  期:
** 描  述:   人眼检测 
*××  
**
** 版  本:   1.0.0
** 备  注:
**
**调用流程：
**  1.使用算法之前
**    初始化EyeInit(...)
**  2.while(需要调用人眼检测算法)
**    {
**        EyeDetection(...)
**    }
**  3.不需要使用算法时
**    EyeRelease(...)
**
**  4.二级筛选在上层恰当位置调用
** 注意：
**  (1)初始化 EyeInit函数 仅需要在第一次使用算法前调用。
**  (2)释放函数 EyeRelease函数 仅需要在最后一次使用算法后调用。
**  (3)人眼追踪算法按照上如上1->2->3顺序调用。
*****************************************************************************************/

#ifndef __INCLUDE_EYEAPI_H__
#define __INCLUDE_EYEAPI_H__
#include "Global.h"


#if defined(_MSC_VER) || defined(WIN32)
#	if defined(_BUILD_EYE_DLL_) && !defined(ASEE_TEST_API)
#		define ASEE_TEST_API __declspec (dllexport) // 实现者应定义该宏（VS中通过预处理器定义）
#		define CALL_EYE __stdcall
#	elif defined(_BUILD_STATIC_EYE_LIB_) && !defined(ASEE_TEST_API)
#		define ASEE_TEST_API  // 实现者应定义该宏（VS中通过预处理器定义）
#		define CALL_EYE 
#	else
#		define ASEE_TEST_API __declspec (dllimport)
#		define CALL_EYE __stdcall
#	endif
#else
#	define CALL_EYE
#	define ASEE_TEST_API // 非windows编译环境
#endif


//人眼类型枚举
typedef enum __EyeType
{
    Neye = 100,
    Leye = 101,
    Reye = 102,
    LrEye = 103,            //不区分左右眼
    LrEyeOpened = 200,      //睁眼
    LrEyeClosed = 201,      //闭眼
}EyeType;


//图像通道枚举
typedef enum __ImgChannel
{
    ImgChannelOne = 1,
    ImgChannelThree = 3,
} ImgChannel;


#include <cstdint>

#ifdef __cplusplus
extern "C"
{
#endif//__cplusplus


	/*****************************************************************************
	*                             初始化
	*  函 数 名：EyeInit
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
    ASEE_TEST_API int32_t CALL_EYE EyeInit(const char* path1, const char* path2, void*& pDevParam, void*& pStatusParam);


	/*****************************************************************************
	*                       释放资源
	*  函 数 名：EyeRelease
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
    ASEE_TEST_API int32_t CALL_EYE EyeRelease(const void* pDevParam, const void* pStatusParam);


	/*****************************************************************************
	*                       眼睛检测
	*  函 数 名：EyeDetection
	*  功    能：实现眼睛检测的库接口
	*  说    明：
    *  参    数：pUsedImage:图像数据结构体指针
    *            pDetEyes:输出人眼检测算法结果
    *            pDevParam:设备参数指针
    *            pStatusParam:状态参数指针
	*  返 回 值：0 ： 成功，非零 ： 失败
	*  创 建 人：liuh
	*  创建时间：2016-08-24
	*  修 改 人：
	*  修改时间：
	*****************************************************************************/
	ASEE_TEST_API int32_t CALL_EYE EyeDetection(const UsedImageData * const pUsedImage, PDetEyes pDetEyes, const void* pDevParam, const void* pStatusParam);


	/*****************************************************************************
	*                       睁闭眼睛检测
	*  函 数 名：EyeStatusDetection
	*  功    能：实现睁闭眼睛检测的库接口
	*  说    明：
    *  参    数：pUsedImage:图像数据结构体指针
    *            pDetEyes:输出人眼检测算法结果
    *            pDevParam:设备参数指针
    *            pStatusParam:状态参数指针
    *  返 回 值：0 ： 成功，非零 ： 失败
	*  创 建 人：liuh
	*  创建时间：2016-08-24
	*  修 改 人：
	*  修改时间：
	*****************************************************************************/
	ASEE_TEST_API int32_t CALL_EYE EyeStatusDetection(const UsedImageData * const pUsedImage, PDetEyes pDetEyes, const void* pDevParam = NULL, const void* pStatusParam = NULL);
    
    
    /*****************************************************************************
	*                       更新参数
	*  函 数 名：EyeUpdate
	*  功    能：更新参数
	*  说    明：暂时没用
	*  参    数：pStatusParam:状态参数指针
	*  返 回 值：0 ： 成功，非零 ： 失败
	*  创 建 人：liuh
	*  创建时间：2016-08-24
	*  修 改 人：
	*  修改时间：
	*****************************************************************************/
    ASEE_TEST_API int32_t CALL_EYE EyeUpdate(const void* pStatusParam);


    /*****************************************************************************
    *                       二级筛选
    *  函 数 名：EyeSecFilter
    *  功    能：二级筛选
    *  说    明：通过二级筛选算法，将pPupilGlintData数组中的置信度进行重新赋值
    *  参    数：pDetEyes:输出人眼检测算法结果
    *            pPupilGlintData:瞳孔光斑计算结果
    *  返 回 值：0 ： 成功，非零 ： 失败
    *  创 建 人：liuh
    *  创建时间：2016-08-24
    *  修 改 人：
    *  修改时间：
    *****************************************************************************/
    ASEE_TEST_API int32_t CALL_EYE EyeSecFilter(const PDetEyes pDetEyes, const PPupilGlintData pPupilGlintData);
	
	/*****************************************************************************
	*                             人眼跟踪初始化
	*  函 数 名：EyeTrackingInit
	*  功    能：人眼跟踪算法初始化
	*  说    明：
	*  参    数：path1:在darknet算法中，path1为cfg文件路径
	*            path2:在darknet算法中，path2为weight文件路径
	*            pDevParam:设备相关变量指针
    *            pStatusParam:状态相关变量指针
	*  返 回 值：0 ： 成功，非零 ： 失败
	*  创 建 人：rendc
	*  创建时间：2016-08-31
	*  修 改 人：
	*  修改时间：
	*****************************************************************************/
	ASEE_TEST_API int32_t CALL_EYE EyeTrackingInit(const char* path1, const char* path2, void*& pDevParam, void*& pStatusParam);


	/*****************************************************************************
	*                       人眼跟踪释放资源
	*  函 数 名：EyeTrackingRelease
	*  功    能：释放眼睛跟踪库使用的相关资源
	*  说    明：
	*  参    数：pDevParam:设备变量指针
    *            pStatusParam:状态变量指针
	*  返 回 值： 0 ： 成功，非零 ： 失败
	*  创 建 人：rendc
	*  创建时间：2016-08-31
	*  修 改 人：
	*  修改时间：
	*****************************************************************************/
    ASEE_TEST_API int32_t CALL_EYE EyeTrackingRelease(const void* pDevParam, const void* pStatusParam);


	/*****************************************************************************
	*                       眼睛跟踪
	*  函 数 名：EyeTracking
	*  功    能：眼睛跟踪的库接口（仅依据人眼区域信息，未依据瞳孔中心、光斑信息）
	*  说    明：
    *  参    数：pDevParam:设备参数指针（包含图像的width,height,color等信息）
	*            ptd:综合下面部分的一个数据结构体：注意：状态参数调整到外面单独使用
	*				 pLastiamge: 上一帧图像的指针
	*				 pNowImage: 本帧图像的指针
    *				 pLastDetEyes: 上一帧人眼检测/跟踪得到的区域
	*				 pNowDetEyes: 本帧人眼跟踪得到的区域 
    *            pStatusParam:状态参数指针
	*  返 回 值：0 ： 成功，非零 ： 失败
	*  创 建 人：rendc
	*  创建时间：2016-08-31
	*  修 改 人：
	*  修改时间：
	*****************************************************************************/
	ASEE_TEST_API int32_t CALL_EYE EyeTracking(const void* pDevParam, const PFDTrackData ptd,const void *pStatusParam);
#ifdef __cplusplus
}//extern "C"
#endif//__cplusplus


#endif//__INCLUDE_EYEAPI_H__