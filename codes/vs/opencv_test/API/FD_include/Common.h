/******************************************************************************************
** 文件名:  Common.h
×× 主要类:
**
** Copyright (c) 七鑫易维有限公司
** 创建人:fjf
** 日  期:2016-08-23
** 修改人:
** 日  期:
** 描  述:对SDK和算法保持最小开放接口拆分出来
*××
**
** 版  本:   1.0.0
** 备  注:通用头文件
**
*****************************************************************************************/
#pragma once
#include <cstdint>
#include <string>

//以下CalibrationGaze.h 中的内容
#ifndef CALIBRATIONGAZE_H
#define CALIBRATIONGAZE_H

// 目前只支持VC和GCC编译器
#if !defined(_MSC_VER) && !defined(__GNUC__)
#	error Unsupported C++ Compiler!!!
#endif

#if defined(WIN32) || defined(_WIN32)
#	define GAZECALL __stdcall
//  Windows平台上的gcc编译器支持“__declspec(dllimport)”和“__declspec(dllexport)”
//#	ifdef __GAZE_DLL__
//#		define GAZEAPI __declspec(dllexport)
//#	elif !defined(__GAZE_STUB__)
//#		define GAZEAPI __declspec(dllimport)
//#	else
#		define GAZEAPI
//#	endif
#else
#	define GAZECALL __attribute__ ((visibility ("default")))
#	define GAZEAPI
#endif

#define GAZE_VER "1611109TMP"

namespace CalibrationGaze
{

	static const int nglints = 8;

	struct PointF
	{
		float x, y;
	};

	struct EyeGlint
	{
		float x, y, re;
	};

	struct EyeFeature
	{
		float pupilx;
		float pupily;
		EyeGlint glints[nglints];

		PointF gaze;

		float opt;
	};

	struct vestPara
	{
		double kappa[2];
		double R;
		double K;
		double backoff;
	};

	struct EyeCalCoe
	{
		double calCoeff[12];
		double auxCoeff[12];
		vestPara vp;
		double dc[12];
		char ver[32];
	};


	// 单个点采集的所有帧数据（从中选取符合要求的10帧）
	struct gazeSelectData
	{
		int totalNum;
		int selectNum;

		EyeFeature eyes[100];
	};

	// 所有点采集的帧数据（用于计算标定系数）
	struct gazeCalData
	{
		int frameNum;		//#Frame selected for each cal Point on the screen.
		int calNum;         //Point number 
		bool finalCal;		// true if this is the final decision.

		EyeFeature	eyes[250];

		EyeCalCoe	coeff; //Output
	};

	// 根据眼睛特征计算注视点的数据
	struct gazeEstData
	{
		EyeFeature	eye;
		EyeCalCoe	coeff;
	};

} //namespace CalibrationGaze

#endif //CALIBRATIONGAZE_H
//以上CalibrationGaze.h 中的内容

typedef  unsigned char BYTE8;

//返回的数据
typedef struct __FeatureData
{
	int reliablity;                     // 置信度， 0-无效
	//眼睛区域
	int RectX;                          //左上角的坐标和宽、高
	int RectY;
	int Width;
	int Height;

	//
	float PupilDiameter;		        // 瞳孔直径
	float PupilX;                       //瞳孔的坐标点X
	float PupilY;                       //瞳孔的坐标点Y
	//Coordinate CoArray[GlintCount];   //光斑点数量
	//分拆光斑点坐标
	float GlintX0;                      //第一个光斑点坐标
	float GlintY0;
	float GlintX1;                      //第二个光斑点坐标 
	float GlintY1;
}FeatureData, *PFeatureData;

//眼睛检测的结果
typedef struct __EyeCheckResult
{
	int CountEye;         //眼睛数量,为眼睛框的个数 
	int LeftFlag;         //左眼检测标记0：未检测到 1：检测到
	int RigthFlag;
	int AllFlag;          //双眼检测到标志 1:检测到双眼
}EyeCheckResult, *PEyeCheckResult;

//双眼的数据结果
typedef struct __FeatureResult
{
	int Status;                //检测状态，即检测到眼睛其则为1，如果为0可以就不处理
	FeatureData LeftData;
	FeatureData RightData;
	EyeCheckResult Result;     //眼睛检测的结果
	long TimeStamp;            //时间戳
	//yrr 2016-11-17线程更新修改
	int eyePupilReCount;            //眼睛和瞳孔都对的个数
}FeatureResult, *PFeatureResult;

//回调函数，用来处理数据回传
typedef void(*CBFeatureOut)(FeatureResult fr);

//错误代码，陆续补充
enum class ResultCode
{
	RESULT_OK = 1,
	RESULT_EER = -1,
};
//配置文件的路径结构体
//在adaboost算法中，path1为xml文件路径
//在darknet算法中，path1为cfg文件路径
//path2 : 在adaboost算法中，path2为model文件路径
//在darknet算法中，path2为weight文件路径
typedef struct __ConfigPath
{
	std::string EyePath1;      //眼睛检测的路径配置文件
	std::string EyePath2;
	std::string TrackPath1;    //跟踪检测的路径配置文件
	std::string TrackPath2;
	std::string PupilPath1;    //瞳孔定位的路径配置文件
	std::string PupilPath2;
}ConfigPath;
//图像传输控制结构体
typedef struct __ImageData
{
	BYTE8 * Buf;          //图像数据流
	short Width;          //图像的宽度
	short Height;         //图像的高度
	int Color;            //位深
	long TimeStamp;       //时间戳
}ImageData, *PImageData;

enum class GlassesStyle
{
	None,                // 无
	Frame,               // 框架
	ContactLens,         // 隐形
	ColoredContactLenses // 美瞳
};
//设备类型
enum DevType
{
	A2_SDK = 0,
	A5_SDK = 1,
	ZTE_PHONE = 2,
	HTC_M8 = 3,
	ASEE_PRO2 = 4,
	A3_SDK=5,
};

//检测模式
enum class CheckMode :short
{
	Left      = 1,    //左眼
	Right     = 2,    //右眼
	Both      = 3,    //双眼
	All = 4,          //混合
};
//图像数据来源枚举
enum class ImageType :short
{
	Camera = 0,    //摄像头
	Pic = 1,       // 图片
	Video = 3,     // 视频
};
//设备参数
typedef struct __DevTypePara
{
	char DevName[256];       //设备名称
	double Fc;               //相机参数
	int MaxDistince;
	int MinDistince;
	int TypiclDistince;

	int Width;
	int Height;
	float GlintDistince;    //光斑距离
	DevType DType;
	CheckMode Mode;         //检测模式（原来的跟踪模式）
	int PupilGlintMode;     //瞳孔方法选择-上层SDK传入
	bool BigEndian;

}DevTypePara, *PDevTypePara;

//定义输入输出宏控制标记
#define IN
#define OUT