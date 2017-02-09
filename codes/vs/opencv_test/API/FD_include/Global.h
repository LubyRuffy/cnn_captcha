/******************************************************************************************
** �ļ���:  Global.h
���� ��Ҫ��:
**
** Copyright (c) ������ά���޹�˾
** ������:fjf
** ��  ��:2016-08-23
** �޸���:
** ��  ��:
** ��  ��:  ��API�ڲ�ʹ�� 
*����
**
** ��  ��:   1.0.0
** ��  ע:ȫ��ͷ�ļ�
**
*****************************************************************************************/

#pragma once
#include <vector>
#include "Common.h"

//��ߵ������
const int GlintCount  = 2;  
const int MaxEyeCount = 32;             //���۸��� 
const int SaveCount   = 20;             //����ͼ�����
const int CheckNum    = 60;             //�ɹ�����۾���������60֡������Check��������Detect�����ߺ�Ϊһ��
const int MaxTrackCache = 20;           //���ٻ���������
const int MaxTrackingEyeCount = 2;      //�������������� Ӧ��ѭ�����Σ��������˵�һֻû�ۣ��ڶ�ֻ���۵���� 

//ͫ�׶�λ��ʹ�õĵ�ṹ��
typedef struct __Point2D32f
{
	float X;
	float Y;
}Point2D3f, *PPoint2D3f;

//ͫ�׶�λ��ʹ�õĵ�ṹ��
typedef struct __Point2DRe
{	
	float X;
	float Y;
	float Re;   //���Ŷȡ��ɹ�Ϊ1��ʧ��Ϊ0
	float Diameter;// yrr 2016-12-07 ֱ��
}Point2DRe, *PPoint2DRe;

//���ο�ṹ��
typedef struct __Rect4f
{
	float X;				//�����������Ͻǹ�һ������x
	float Y;				//�����������Ͻǹ�һ������y
	float Width;            //���������һ�����w
	float Height;           //���������һ���߶�h
}Rect4f,*PRect4f;
//���ο�ṹ��
typedef struct __Rect4i
{
	int X;				 //�����������Ͻ�����x
	int Y;				 //�����������Ͻ�����y
	int Width;           //����������w
	int Height;          //��������߶�h
}Rect4i, *PRect4i;
//ͫ�׺͹�߷��ص����ݽṹ��
typedef struct __PupilGlintData
{
	Point2DRe Pupil;
	Point2DRe Glints[GlintCount];
}PupilGlintData, *PPupilGlintData;

//����
// ----->x
// |
//\|/
// y
typedef struct __SingleEye
{
	Rect4f Rc4f;            //���������һ��������
	Rect4i Rc4i;            //������������

	int32_t BIsEye;        //�Ƿ�����������1:��  0:����
	int32_t EyeClassify;   //0:δɸѡ 10x:ɸѡ���������� --�����۵���ϸ˵����EyeApi.h�е�ö��˵��
	int32_t EyeStatus;     //200:���� 201:����
	float   Prob;          //���ۿ��Ŷ�

	int32_t FeatureType;   //��������
	int32_t NoUse1;        //�����ӿڣ�����ʹ��ʱʹ��
	int32_t NoUse2;        //�����ӿڣ�����ʹ��ʱʹ��
}SingleEye, *PSingleEye;


typedef struct __DetEyes
{
	SingleEye ResDetEyes[MaxEyeCount]; //���ؼ����������
	int       ResDetEyesCount;         //���ؼ����������
}DetEyes, *PDetEyes;

//��ʵ��ʹ�õ�����
typedef struct __UsedImageData
{
	ImageData IData;
	void * matData;
	BYTE8 * matBuf;            //ָ��Mat�ڲ�BYTE*��ָ��
	Rect4f eyeRect;
}UsedImageData, *PUsedImageData;

//��������
typedef struct __FDTrackData
{
	PupilGlintData * LastData;             //��һ֡��ͫ�׺͹�߶�λ���
	DetEyes LastDetEyes;                   //��һ֡���۾������
	DetEyes CurDetEyes;                    //��ǰ׷�ٷ��صĽ��
	UsedImageData * LastImage;             //��һ֡ͼ��--��ͼ 
	UsedImageData * CurImage;              //��ǰ֡ͼ��--��ͼ
	//void *StatusParam;                   //״̬����-�������¸��ٺ��״̬-�޸ĵ������ڲ�
	int CurStatus;                         //0:track 1:detect
}FDTrackData,*PFDTrackData;


//�ж��۾�ͫ��λ��Ľ��״̬���ݽṹ��
typedef struct __EyeResultStatus
{
	bool IsSingle;    // �Ƿ���
	int EyeType;      //0:˫��  1������  2������ 3��δ��⵽
}EyeResultStatus, PEyeResultStatus;

//�۾��ļ�⵱ǰ����״̬
enum class EyeCurStatus
{
	None = 0,       //˫���Ҳ���ʱ������
	Single = 1,     //����۵ĵ�˫��
	Both = 2,
};
//��ǰ������״̬
enum class CurRunStatus
{
	Track = 0,            //����״̬
	Check = 1,            //���״̬
	Detect = 2,           //���״̬
};





