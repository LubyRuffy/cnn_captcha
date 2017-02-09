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



enum OCR_ERROR
{
	OCR_OK = 0,
    OCR_ERROR_OVERFLOW = 1,
	OCR_ERROR_PARAMER = 2,
	OCR_ERROR_CHANNELS = 3,
    OCR_ERROR_TYPE = 4,
};
#define OCR_RET(A) (int32_t)(A)


//ͼ������ƽṹ��
typedef struct __ImageData
{
    unsigned char * Buf;  //ͼ��������
	short Width;          //ͼ��Ŀ��
	short Height;         //ͼ��ĸ߶�
	int   Channels;       //ͼ��ͨ����
}ImageData, *PImageData;


//ͼ������ƽṹ��
typedef struct __RetStr
{
    unsigned char RetBuf[16];   //�������� ��෵��10��
    int           RetCnt;
}RetStr, *PRetStr;


#ifdef __cplusplus
extern "C"
{
#endif//__cplusplus


	/*****************************************************************************
    *                       OCR
    *  �� �� ���� OcrRecognition
    *  ˵    ����
    *  ��    ����PImageData:ͼ������
    *            pPRetStr:���ocr=���
    *            ocr_type:ocr����
    *                     Ӣ�����ֻ�� ocr_type=3000 ���ⳤ��Ӣ����ϣ�ʶ���ʻή��
    *                     ocr_type=3010 1λӢ��
    *                     ocr_type=3020 2λӢ�����
    *                     ocr_type=3100 10λӢ�����
	*  �� �� ֵ��0 �� �ɹ������� �� ʧ��
	*****************************************************************************/
    APP_TEST_API int32_t __stdcall OcrRecognition(const ImageData * const PImageData, PRetStr pPRetStr, int ocr_type);

#ifdef __cplusplus
}//extern "C"
#endif//__cplusplus


#endif//__INCLUDE_OCR_API_H__