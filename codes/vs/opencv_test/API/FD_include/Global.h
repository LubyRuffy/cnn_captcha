/******************************************************************************************
** 文件名:  Global.h
×× 主要类:
**
** Copyright (c) 七鑫易维有限公司
** 创建人:fjf
** 日  期:2016-08-23
** 修改人:
** 日  期:
** 描  述:  供API内部使用 
*××
**
** 版  本:   1.0.0
** 备  注:全局头文件
**
*****************************************************************************************/

#pragma once
#include <vector>
#include "Common.h"

//光斑点的数量
const int GlintCount  = 2;  
const int MaxEyeCount = 32;             //人眼个数 
const int SaveCount   = 20;             //保存图像个数
const int CheckNum    = 60;             //成功检测眼睛的数量到60帧后，启动Check（即启动Detect，二者合为一）
const int MaxTrackCache = 20;           //跟踪缓存最大个数
const int MaxTrackingEyeCount = 2;      //跟踪人眼最大个数 应该循环两次，否则处理不了第一只没眼，第二只有眼的情况 

//瞳孔定位中使用的点结构体
typedef struct __Point2D32f
{
	float X;
	float Y;
}Point2D3f, *PPoint2D3f;

//瞳孔定位中使用的点结构体
typedef struct __Point2DRe
{	
	float X;
	float Y;
	float Re;   //置信度。成功为1，失败为0
	float Diameter;// yrr 2016-12-07 直径
}Point2DRe, *PPoint2DRe;

//矩形框结构体
typedef struct __Rect4f
{
	float X;				//矩形区域左上角归一化坐标x
	float Y;				//矩形区域左上角归一化坐标y
	float Width;            //矩形区域归一化宽度w
	float Height;           //矩形区域归一化高度h
}Rect4f,*PRect4f;
//矩形框结构体
typedef struct __Rect4i
{
	int X;				 //矩形区域左上角坐标x
	int Y;				 //矩形区域左上角坐标y
	int Width;           //矩形区域宽度w
	int Height;          //矩形区域高度h
}Rect4i, *PRect4i;
//瞳孔和光斑返回的数据结构体
typedef struct __PupilGlintData
{
	Point2DRe Pupil;
	Point2DRe Glints[GlintCount];
}PupilGlintData, *PPupilGlintData;

//坐标
// ----->x
// |
//\|/
// y
typedef struct __SingleEye
{
	Rect4f Rc4f;            //人眼区域归一化后坐标
	Rect4i Rc4i;            //人眼区域坐标

	int32_t BIsEye;        //是否是人眼区域，1:是  0:不是
	int32_t EyeClassify;   //0:未筛选 10x:筛选出来的人眼 --左右眼的详细说明看EyeApi.h中的枚举说明
	int32_t EyeStatus;     //200:睁眼 201:闭眼
	float   Prob;          //人眼可信度

	int32_t FeatureType;   //特征类型
	int32_t NoUse1;        //保留接口，后续使用时使用
	int32_t NoUse2;        //保留接口，后续使用时使用
}SingleEye, *PSingleEye;


typedef struct __DetEyes
{
	SingleEye ResDetEyes[MaxEyeCount]; //返回检测人眼数据
	int       ResDetEyesCount;         //返回检测人眼数量
}DetEyes, *PDetEyes;

//在实际使用的数据
typedef struct __UsedImageData
{
	ImageData IData;
	void * matData;
	BYTE8 * matBuf;            //指向Mat内部BYTE*的指针
	Rect4f eyeRect;
}UsedImageData, *PUsedImageData;

//跟踪数据
typedef struct __FDTrackData
{
	PupilGlintData * LastData;             //上一帧的瞳孔和光斑定位结果
	DetEyes LastDetEyes;                   //上一帧的眼睛检测结果
	DetEyes CurDetEyes;                    //当前追踪返回的结果
	UsedImageData * LastImage;             //上一帧图像--大图 
	UsedImageData * CurImage;              //当前帧图像--大图
	//void *StatusParam;                   //状态参数-用来更新跟踪后的状态-修改到函数内部
	int CurStatus;                         //0:track 1:detect
}FDTrackData,*PFDTrackData;


//判断眼睛瞳定位后的结果状态数据结构体
typedef struct __EyeResultStatus
{
	bool IsSingle;    // 是否单眼
	int EyeType;      //0:双眼  1：左眼  2：右眼 3：未检测到
}EyeResultStatus, PEyeResultStatus;

//眼睛的检测当前数量状态
enum class EyeCurStatus
{
	None = 0,       //双眼找不到时，单眼
	Single = 1,     //混合眼的单双眼
	Both = 2,
};
//当前的运行状态
enum class CurRunStatus
{
	Track = 0,            //跟踪状态
	Check = 1,            //检查状态
	Detect = 2,           //检测状态
};





