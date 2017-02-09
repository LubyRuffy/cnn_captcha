/******************************************************************************************
** �ļ���:  OCR_Api.h
���� ��Ҫ��:    
**
** Copyright (c) 
** ������:
** ��  ��:
** �޸���:
** ��  ��:
** ��  ��:   OCR_Api
*����  
**
** ��  ��:   1.0.0
** ��  ע:
**
*****************************************************************************************/

#ifndef __INCLUDE_OCR_API_H__
#define __INCLUDE_OCR_API_H__


#if defined(_MSC_VER) || defined(WIN32)
#	if defined(_BUILD_OCR_DLL_) && !defined(APP_TEST_API)
#		define APP_TEST_API __declspec (dllexport) // ʵ����Ӧ����ú꣨VS��ͨ��Ԥ���������壩
#		define CALL_OCR __stdcall
#	elif defined(_BUILD_STATIC_OCR_LIB_) && !defined(APP_TEST_API)
#		define APP_TEST_API  // ʵ����Ӧ����ú꣨VS��ͨ��Ԥ���������壩
#		define CALL_OCR 
#	else
#		define APP_TEST_API __declspec (dllimport)
#		define CALL_OCR __stdcall
#	endif
#else
#	define CALL_OCR
#	define APP_TEST_API // ��windows���뻷��
#endif


#include <cstdint>

#ifdef __cplusplus
extern "C"
{
#endif//__cplusplus


	/*****************************************************************************
	*                             ��ʼ��
	*  �� �� ����OcrInit
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
    APP_TEST_API int32_t CALL_OCR OcrInit(const char* path1, const char* path2, void*& pDevParam, void*& pStatusParam);


	/*****************************************************************************
	*                       �ͷ���Դ
	*  �� �� ����OcrRelease
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
    APP_TEST_API int32_t CALL_OCR OcrRelease(const void* pDevParam, const void* pStatusParam);


	/*****************************************************************************
	*                       �۾����
	*  �� �� ����OcrDetection
	*  ��    �ܣ�ʵ���۾����Ŀ�ӿ�
	*  ˵    ����
    *  ��    ����pUsedImage:ͼ�����ݽṹ��ָ��
    *            pDetOcrs:������ۼ���㷨���
    *            pDevParam:�豸����ָ��
    *            pStatusParam:״̬����ָ��
	*  �� �� ֵ��0 �� �ɹ������� �� ʧ��
	*  �� �� �ˣ�liuh
	*  ����ʱ�䣺2016-08-24
	*  �� �� �ˣ�
	*  �޸�ʱ�䣺
	*****************************************************************************/
	APP_TEST_API int32_t CALL_OCR OcrDetection(const UsedImageData * const pUsedImage, PDetOcrs pDetOcrs, const void* pDevParam, const void* pStatusParam);

#ifdef __cplusplus
}//extern "C"
#endif//__cplusplus


#endif//__INCLUDE_OCR_API_H__