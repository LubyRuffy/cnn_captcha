
/******************************************************************************************
** 文件名:  IPupilGlintCompute.h
×× 主要类:
**
** Copyright (c) 七鑫易维有限公司
** 创建人:lwb
** 日  期:2016-08-22
** 修改人:Will
** 日  期:2016-9-9
** 描  述:   瞳孔定位和光斑位置
*××
**
** 版  本:   1.0.1
** 备  注: 
*       s1: PupilGlintInitDevParam
*       s2: PupilGlintInitStatusParam
*       s3: PupilGlintInitTempData
*       s3.1 PupilGlintClipEye
*       s4: PupilGlintComputeCenter
		s4.1 PupilGlintReleaseEye
*       s5: PupilGlintUpdate
*       s6: PupilGlintReleaseTempData
*       s7: goto s3
*       s8: PupilGlintReleaseStatusParam
**
*****************************************************************************************/

#pragma once
#include "Global.h"

/*****************************************************************************
*                       初始化设备相关参数
*  函 数 名：PupilGlintInitDevParam
*  功    能：初始化设备参数
*  说    明：
*
*  参    数： dev:设备类型参数
*             pDevParam：设备相关参数
*             path1:cfg文件路径
*             path2:weights文件路径
*  返 回 值：成功为0，失败为非0错误码
*  创 建 人：
*  创建时间：2016-08-22
*  修 改 人：
*  修改时间：
*****************************************************************************/
int32_t PupilGlintInitDevParam(DevTypePara dev, void*& pDevParam, const char* path1, const char* path2);

/*****************************************************************************
*                       初始化状态更新参数
*  函 数 名：PupilGlintInitStatusParam
*  功    能：初始化状态参数
*  说    明：
*
*  参    数： dev:设备类型参数
*             pStatusParam：状态参数
*  返 回 值：成功为0，失败为非0错误码
*  创 建 人：
*  创建时间：2016-08-22
*  修 改 人：
*  修改时间：
*****************************************************************************/
int32_t PupilGlintInitStatusParam( DevTypePara dev, void*& pStatusParam);

/*****************************************************************************
*                       初始化临时数据
*  函 数 名：PupilGlintInitTempData
*  功    能：初始化临时数据
*  说    明：
*
*  参    数： dev:设备类型参数
*             pTempData：更新状态参数时需要使用的临时参数
*  返 回 值：成功为0，失败为非0错误码
*  创 建 人：
*  创建时间：2016-08-22
*  修 改 人：
*  修改时间：
*****************************************************************************/
int32_t PupilGlintInitTempData(DevTypePara dev, void*& pTempData);

/*****************************************************************************
*                       检测瞳孔光斑
*  函 数 名：ComputePupilGlintCenter
*  功    能：
*  说    明：注意：如果在操作瞳孔正确而光斑有问题的情况下，要把置信度置成0，防止出现
*                  二者冲突对算法的冲击。
*  参    数： nFrameNum：当前帧编号，如果区分左右眼，需要各自编号
*             pImgEye：眼睛图像，包含位置信息，自定义类型，由PupilGlintClipEye返回
*             pupilData：返回的瞳孔和光斑位置  !!在原图中的位置!!
*             pTempData：中间变量
*             pDevParam：传入的设备相关参数
*             pStatusParam：状态参数
*  返 回 值：成功为true，失败为false
*  创 建 人：
*  创建时间：2016-9-9
*  修 改 人：Will
*  修改时间：
*****************************************************************************/
bool PupilGlintComputeCenter(
	int nFrameNum,
    const void* const pImgEye,
	PupilGlintData &pupilData,
	void * pTempData = NULL,//自定义内容传给Update函数
	const void * const pDevParam = NULL, 
	const void * const pStatusParam = NULL
	);

/*****************************************************************************
*                       释放状态参数
*  函 数 名：PupilGlintReleaseStatusParam
*  功    能：
*  说    明：
*
*  参    数：
*           pStatusParam ：状态参数
*  返 回 值：
*  创 建 人：
*  创建时间：2016-08-19
*  修 改 人：
*  修改时间：
*****************************************************************************/
void PupilGlintReleaseStatusParam(void * pStatusParam);

/*****************************************************************************
*                       释放临时数据
*  函 数 名：PupilGlintReleaseTempData
*  功    能：
*  说    明：
*
*  参    数：
*           pTempData ：临时数据
*  返 回 值：
*  创 建 人：
*  创建时间：2016-08-19
*  修 改 人：
*  修改时间：
*****************************************************************************/
void PupilGlintReleaseTempData(void * pTempData);

/*****************************************************************************
*                       更新状态参数
*  函 数 名：PupilGlintUpdate
*  功    能：
*  说    明：
*
*  参    数：
*          pTempData：临时中间数据
*          pStatusParam ：由上面的pTempData更新本参数
*  返 回 值：
*  创 建 人：
*  创建时间：2016-08-19
*  修 改 人：
*  修改时间：
*****************************************************************************/
void PupilGlintUpdate(void * pTempData,void* pStatusParam);

/*****************************************************************************
*                       眼睛截图
*  函 数 名：PupilGlintClipEye
*  功    能：重截取眼睛区域，部分算法需要使瞳孔近似居中
*  说    明：
*
*  参    数：
*          pImgIn：输入的图像
*          rec ：裁剪大小
*          pImgEye:输出的图像，及其在原图中的位置

*  返 回 值：成功为0，失败为非0错误码
*  创 建 人：
*  创建时间：2016-09-09
*  修 改 人：Will
*  修改时间：
*****************************************************************************/
int32_t PupilGlintClipEye(const void* pStatusParam, const UsedImageData* pImgIn, Rect4i rec, void *& pImgEye);

/*****************************************************************************
*                       释放眼睛截图的内存
*  函 数 名：PupilGlintReleaseEye
*  功    能：释放 imgEye
*  说    明：
*
*  参    数：

*          imgEye:眼睛的图像

*  返 回 值：
*  创 建 人：
*  创建时间：2016-09-09
*  修 改 人：Will
*  修改时间：
*****************************************************************************/
void PupilGlintReleaseEye(void *& pImgEye);

/*****************************************************************************
*                      DETECT和TRACK间的状态复制
*  函 数 名：PupilGlintStatusCopy
*  功    能：
*  说    明：
*
*  参    数：
*          trackStatus：输出的状态：跟踪状态参数
*          
*          detectStatus:输入的状态：检测状态参数
*  返 回 值：
*  创 建 人：
*  创建时间：2016-09-02
*  修 改 人：
*  修改时间：
*****************************************************************************/
void PupilGlintStatusCopy(OUT void * trackStatus, IN void* detectStatus); 
