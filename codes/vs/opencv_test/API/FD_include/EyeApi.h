/******************************************************************************************
** �ļ���:  EyeApi.h
���� ��Ҫ��:    
**
** Copyright (c) ������ά���޹�˾
** ������:liuh
** ��  ��:2016-08-24
** �޸���:
** ��  ��:
** ��  ��:   ���ۼ�� 
*����  
**
** ��  ��:   1.0.0
** ��  ע:
**
**�������̣�
**  1.ʹ���㷨֮ǰ
**    ��ʼ��EyeInit(...)
**  2.while(��Ҫ�������ۼ���㷨)
**    {
**        EyeDetection(...)
**    }
**  3.����Ҫʹ���㷨ʱ
**    EyeRelease(...)
**
**  4.����ɸѡ���ϲ�ǡ��λ�õ���
** ע�⣺
**  (1)��ʼ�� EyeInit���� ����Ҫ�ڵ�һ��ʹ���㷨ǰ���á�
**  (2)�ͷź��� EyeRelease���� ����Ҫ�����һ��ʹ���㷨����á�
**  (3)����׷���㷨����������1->2->3˳����á�
*****************************************************************************************/

#ifndef __INCLUDE_EYEAPI_H__
#define __INCLUDE_EYEAPI_H__
#include "Global.h"


#if defined(_MSC_VER) || defined(WIN32)
#	if defined(_BUILD_EYE_DLL_) && !defined(ASEE_TEST_API)
#		define ASEE_TEST_API __declspec (dllexport) // ʵ����Ӧ����ú꣨VS��ͨ��Ԥ���������壩
#		define CALL_EYE __stdcall
#	elif defined(_BUILD_STATIC_EYE_LIB_) && !defined(ASEE_TEST_API)
#		define ASEE_TEST_API  // ʵ����Ӧ����ú꣨VS��ͨ��Ԥ���������壩
#		define CALL_EYE 
#	else
#		define ASEE_TEST_API __declspec (dllimport)
#		define CALL_EYE __stdcall
#	endif
#else
#	define CALL_EYE
#	define ASEE_TEST_API // ��windows���뻷��
#endif


//��������ö��
typedef enum __EyeType
{
    Neye = 100,
    Leye = 101,
    Reye = 102,
    LrEye = 103,            //������������
    LrEyeOpened = 200,      //����
    LrEyeClosed = 201,      //����
}EyeType;


//ͼ��ͨ��ö��
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
	*                             ��ʼ��
	*  �� �� ����EyeInit
	*  ��    �ܣ����ۼ���㷨��ʼ��
	*  ˵    ����
	*  ��    ����path1:��adaboost�㷨�У�path1Ϊxml�ļ�·��
    *                  ��darknet�㷨�У�path1Ϊcfg�ļ�·��
	*            path2:��adaboost�㷨�У�path2Ϊmodel�ļ�·��
    *                  ��darknet�㷨�У�path2Ϊweight�ļ�·��
	*            pDevParam:�豸��ر���ָ��
    *            pStatusParam:״̬��ر���ָ��
	*  �� �� ֵ��0 �� �ɹ������� �� ʧ��
	*  �� �� �ˣ�liuh
	*  ����ʱ�䣺2016-08-24
	*  �� �� �ˣ�
	*  �޸�ʱ�䣺
	*****************************************************************************/
    ASEE_TEST_API int32_t CALL_EYE EyeInit(const char* path1, const char* path2, void*& pDevParam, void*& pStatusParam);


	/*****************************************************************************
	*                       �ͷ���Դ
	*  �� �� ����EyeRelease
	*  ��    �ܣ��ͷ��۾�����ʹ�õ������Դ
	*  ˵    ����
	*  ��    ����pDevParam:�豸����ָ��
    *            pStatusParam:״̬����ָ��
	*  �� �� ֵ�� 0 �� �ɹ������� �� ʧ��
	*  �� �� �ˣ�liuh
	*  ����ʱ�䣺2016-08-24
	*  �� �� �ˣ�
	*  �޸�ʱ�䣺
	*****************************************************************************/
    ASEE_TEST_API int32_t CALL_EYE EyeRelease(const void* pDevParam, const void* pStatusParam);


	/*****************************************************************************
	*                       �۾����
	*  �� �� ����EyeDetection
	*  ��    �ܣ�ʵ���۾����Ŀ�ӿ�
	*  ˵    ����
    *  ��    ����pUsedImage:ͼ�����ݽṹ��ָ��
    *            pDetEyes:������ۼ���㷨���
    *            pDevParam:�豸����ָ��
    *            pStatusParam:״̬����ָ��
	*  �� �� ֵ��0 �� �ɹ������� �� ʧ��
	*  �� �� �ˣ�liuh
	*  ����ʱ�䣺2016-08-24
	*  �� �� �ˣ�
	*  �޸�ʱ�䣺
	*****************************************************************************/
	ASEE_TEST_API int32_t CALL_EYE EyeDetection(const UsedImageData * const pUsedImage, PDetEyes pDetEyes, const void* pDevParam, const void* pStatusParam);


	/*****************************************************************************
	*                       �����۾����
	*  �� �� ����EyeStatusDetection
	*  ��    �ܣ�ʵ�������۾����Ŀ�ӿ�
	*  ˵    ����
    *  ��    ����pUsedImage:ͼ�����ݽṹ��ָ��
    *            pDetEyes:������ۼ���㷨���
    *            pDevParam:�豸����ָ��
    *            pStatusParam:״̬����ָ��
    *  �� �� ֵ��0 �� �ɹ������� �� ʧ��
	*  �� �� �ˣ�liuh
	*  ����ʱ�䣺2016-08-24
	*  �� �� �ˣ�
	*  �޸�ʱ�䣺
	*****************************************************************************/
	ASEE_TEST_API int32_t CALL_EYE EyeStatusDetection(const UsedImageData * const pUsedImage, PDetEyes pDetEyes, const void* pDevParam = NULL, const void* pStatusParam = NULL);
    
    
    /*****************************************************************************
	*                       ���²���
	*  �� �� ����EyeUpdate
	*  ��    �ܣ����²���
	*  ˵    ������ʱû��
	*  ��    ����pStatusParam:״̬����ָ��
	*  �� �� ֵ��0 �� �ɹ������� �� ʧ��
	*  �� �� �ˣ�liuh
	*  ����ʱ�䣺2016-08-24
	*  �� �� �ˣ�
	*  �޸�ʱ�䣺
	*****************************************************************************/
    ASEE_TEST_API int32_t CALL_EYE EyeUpdate(const void* pStatusParam);


    /*****************************************************************************
    *                       ����ɸѡ
    *  �� �� ����EyeSecFilter
    *  ��    �ܣ�����ɸѡ
    *  ˵    ����ͨ������ɸѡ�㷨����pPupilGlintData�����е����ŶȽ������¸�ֵ
    *  ��    ����pDetEyes:������ۼ���㷨���
    *            pPupilGlintData:ͫ�׹�߼�����
    *  �� �� ֵ��0 �� �ɹ������� �� ʧ��
    *  �� �� �ˣ�liuh
    *  ����ʱ�䣺2016-08-24
    *  �� �� �ˣ�
    *  �޸�ʱ�䣺
    *****************************************************************************/
    ASEE_TEST_API int32_t CALL_EYE EyeSecFilter(const PDetEyes pDetEyes, const PPupilGlintData pPupilGlintData);
	
	/*****************************************************************************
	*                             ���۸��ٳ�ʼ��
	*  �� �� ����EyeTrackingInit
	*  ��    �ܣ����۸����㷨��ʼ��
	*  ˵    ����
	*  ��    ����path1:��darknet�㷨�У�path1Ϊcfg�ļ�·��
	*            path2:��darknet�㷨�У�path2Ϊweight�ļ�·��
	*            pDevParam:�豸��ر���ָ��
    *            pStatusParam:״̬��ر���ָ��
	*  �� �� ֵ��0 �� �ɹ������� �� ʧ��
	*  �� �� �ˣ�rendc
	*  ����ʱ�䣺2016-08-31
	*  �� �� �ˣ�
	*  �޸�ʱ�䣺
	*****************************************************************************/
	ASEE_TEST_API int32_t CALL_EYE EyeTrackingInit(const char* path1, const char* path2, void*& pDevParam, void*& pStatusParam);


	/*****************************************************************************
	*                       ���۸����ͷ���Դ
	*  �� �� ����EyeTrackingRelease
	*  ��    �ܣ��ͷ��۾����ٿ�ʹ�õ������Դ
	*  ˵    ����
	*  ��    ����pDevParam:�豸����ָ��
    *            pStatusParam:״̬����ָ��
	*  �� �� ֵ�� 0 �� �ɹ������� �� ʧ��
	*  �� �� �ˣ�rendc
	*  ����ʱ�䣺2016-08-31
	*  �� �� �ˣ�
	*  �޸�ʱ�䣺
	*****************************************************************************/
    ASEE_TEST_API int32_t CALL_EYE EyeTrackingRelease(const void* pDevParam, const void* pStatusParam);


	/*****************************************************************************
	*                       �۾�����
	*  �� �� ����EyeTracking
	*  ��    �ܣ��۾����ٵĿ�ӿڣ�����������������Ϣ��δ����ͫ�����ġ������Ϣ��
	*  ˵    ����
    *  ��    ����pDevParam:�豸����ָ�루����ͼ���width,height,color����Ϣ��
	*            ptd:�ۺ����沿�ֵ�һ�����ݽṹ�壺ע�⣺״̬�������������浥��ʹ��
	*				 pLastiamge: ��һ֡ͼ���ָ��
	*				 pNowImage: ��֡ͼ���ָ��
    *				 pLastDetEyes: ��һ֡���ۼ��/���ٵõ�������
	*				 pNowDetEyes: ��֡���۸��ٵõ������� 
    *            pStatusParam:״̬����ָ��
	*  �� �� ֵ��0 �� �ɹ������� �� ʧ��
	*  �� �� �ˣ�rendc
	*  ����ʱ�䣺2016-08-31
	*  �� �� �ˣ�
	*  �޸�ʱ�䣺
	*****************************************************************************/
	ASEE_TEST_API int32_t CALL_EYE EyeTracking(const void* pDevParam, const PFDTrackData ptd,const void *pStatusParam);
#ifdef __cplusplus
}//extern "C"
#endif//__cplusplus


#endif//__INCLUDE_EYEAPI_H__