/******************************************************************************************
** �ļ���:  Common.h
���� ��Ҫ��:
**
** Copyright (c) ������ά���޹�˾
** ������:fjf
** ��  ��:2016-08-23
** �޸���:
** ��  ��:
** ��  ��:��SDK���㷨������С���Žӿڲ�ֳ���
*����
**
** ��  ��:   1.0.0
** ��  ע:ͨ��ͷ�ļ�
**
*****************************************************************************************/
#pragma once
#include <cstdint>
#include <string>

//����CalibrationGaze.h �е�����
#ifndef CALIBRATIONGAZE_H
#define CALIBRATIONGAZE_H

// Ŀǰֻ֧��VC��GCC������
#if !defined(_MSC_VER) && !defined(__GNUC__)
#	error Unsupported C++ Compiler!!!
#endif

#if defined(WIN32) || defined(_WIN32)
#	define GAZECALL __stdcall
//  Windowsƽ̨�ϵ�gcc������֧�֡�__declspec(dllimport)���͡�__declspec(dllexport)��
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


	// ������ɼ�������֡���ݣ�����ѡȡ����Ҫ���10֡��
	struct gazeSelectData
	{
		int totalNum;
		int selectNum;

		EyeFeature eyes[100];
	};

	// ���е�ɼ���֡���ݣ����ڼ���궨ϵ����
	struct gazeCalData
	{
		int frameNum;		//#Frame selected for each cal Point on the screen.
		int calNum;         //Point number 
		bool finalCal;		// true if this is the final decision.

		EyeFeature	eyes[250];

		EyeCalCoe	coeff; //Output
	};

	// �����۾���������ע�ӵ������
	struct gazeEstData
	{
		EyeFeature	eye;
		EyeCalCoe	coeff;
	};

} //namespace CalibrationGaze

#endif //CALIBRATIONGAZE_H
//����CalibrationGaze.h �е�����

typedef  unsigned char BYTE8;

//���ص�����
typedef struct __FeatureData
{
	int reliablity;                     // ���Ŷȣ� 0-��Ч
	//�۾�����
	int RectX;                          //���Ͻǵ�����Ϳ���
	int RectY;
	int Width;
	int Height;

	//
	float PupilDiameter;		        // ͫ��ֱ��
	float PupilX;                       //ͫ�׵������X
	float PupilY;                       //ͫ�׵������Y
	//Coordinate CoArray[GlintCount];   //��ߵ�����
	//�ֲ��ߵ�����
	float GlintX0;                      //��һ����ߵ�����
	float GlintY0;
	float GlintX1;                      //�ڶ�����ߵ����� 
	float GlintY1;
}FeatureData, *PFeatureData;

//�۾����Ľ��
typedef struct __EyeCheckResult
{
	int CountEye;         //�۾�����,Ϊ�۾���ĸ��� 
	int LeftFlag;         //���ۼ����0��δ��⵽ 1����⵽
	int RigthFlag;
	int AllFlag;          //˫�ۼ�⵽��־ 1:��⵽˫��
}EyeCheckResult, *PEyeCheckResult;

//˫�۵����ݽ��
typedef struct __FeatureResult
{
	int Status;                //���״̬������⵽�۾�����Ϊ1�����Ϊ0���ԾͲ�����
	FeatureData LeftData;
	FeatureData RightData;
	EyeCheckResult Result;     //�۾����Ľ��
	long TimeStamp;            //ʱ���
	//yrr 2016-11-17�̸߳����޸�
	int eyePupilReCount;            //�۾���ͫ�׶��Եĸ���
}FeatureResult, *PFeatureResult;

//�ص������������������ݻش�
typedef void(*CBFeatureOut)(FeatureResult fr);

//������룬½������
enum class ResultCode
{
	RESULT_OK = 1,
	RESULT_EER = -1,
};
//�����ļ���·���ṹ��
//��adaboost�㷨�У�path1Ϊxml�ļ�·��
//��darknet�㷨�У�path1Ϊcfg�ļ�·��
//path2 : ��adaboost�㷨�У�path2Ϊmodel�ļ�·��
//��darknet�㷨�У�path2Ϊweight�ļ�·��
typedef struct __ConfigPath
{
	std::string EyePath1;      //�۾�����·�������ļ�
	std::string EyePath2;
	std::string TrackPath1;    //���ټ���·�������ļ�
	std::string TrackPath2;
	std::string PupilPath1;    //ͫ�׶�λ��·�������ļ�
	std::string PupilPath2;
}ConfigPath;
//ͼ������ƽṹ��
typedef struct __ImageData
{
	BYTE8 * Buf;          //ͼ��������
	short Width;          //ͼ��Ŀ��
	short Height;         //ͼ��ĸ߶�
	int Color;            //λ��
	long TimeStamp;       //ʱ���
}ImageData, *PImageData;

enum class GlassesStyle
{
	None,                // ��
	Frame,               // ���
	ContactLens,         // ����
	ColoredContactLenses // ��ͫ
};
//�豸����
enum DevType
{
	A2_SDK = 0,
	A5_SDK = 1,
	ZTE_PHONE = 2,
	HTC_M8 = 3,
	ASEE_PRO2 = 4,
	A3_SDK=5,
};

//���ģʽ
enum class CheckMode :short
{
	Left      = 1,    //����
	Right     = 2,    //����
	Both      = 3,    //˫��
	All = 4,          //���
};
//ͼ��������Դö��
enum class ImageType :short
{
	Camera = 0,    //����ͷ
	Pic = 1,       // ͼƬ
	Video = 3,     // ��Ƶ
};
//�豸����
typedef struct __DevTypePara
{
	char DevName[256];       //�豸����
	double Fc;               //�������
	int MaxDistince;
	int MinDistince;
	int TypiclDistince;

	int Width;
	int Height;
	float GlintDistince;    //��߾���
	DevType DType;
	CheckMode Mode;         //���ģʽ��ԭ���ĸ���ģʽ��
	int PupilGlintMode;     //ͫ�׷���ѡ��-�ϲ�SDK����
	bool BigEndian;

}DevTypePara, *PDevTypePara;

//���������������Ʊ��
#define IN
#define OUT