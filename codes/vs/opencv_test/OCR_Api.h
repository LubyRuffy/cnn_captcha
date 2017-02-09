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


#if defined(_BUILD_OCR_DLL_) && !defined(APP_TEST_API)
#		define APP_TEST_API __declspec (dllexport) // ʵ����Ӧ����ú꣨VS��ͨ��Ԥ���������壩
#	else
#		define APP_TEST_API __declspec (dllimport)
#endif



#include <cstdint>



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
    unsigned char   RetBuf[16];   //�������� ��෵��10��
    int     RetCnt;
}RetStr, *PRetStr;


#ifdef __cplusplus
extern "C"
{
#endif//__cplusplus


    /*****************************************************************************
    *                       OcrInit
    *  �� �� ���� OcrInit
    *  ˵    ����
    *  ��    ����handle:���
    *            path1:·��1
    *            path2:·��2
    *  �� �� ֵ��0 �� �ɹ������� �� ʧ��
    *****************************************************************************/
    APP_TEST_API int32_t __stdcall OcrInit(const void** handle, const char* path1, const char* path2);


    /*****************************************************************************
    *                       OcrRelease
    *  �� �� ���� OcrRelease
    *  ˵    ����
    *  ��    ����handle:���
    *  �� �� ֵ��0 �� �ɹ������� �� ʧ��
    *****************************************************************************/
    APP_TEST_API int32_t __stdcall OcrRelease(void* handle);

	
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
    APP_TEST_API int32_t __stdcall OcrRecognition(const void* handle, const ImageData * const PImageData, PRetStr pPRetStr, int ocr_type);

#ifdef __cplusplus
}//extern "C"
#endif//__cplusplus


#endif//__INCLUDE_OCR_API_H__