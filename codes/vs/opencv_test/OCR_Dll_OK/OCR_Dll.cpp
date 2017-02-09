// OCR_Dll.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "Includes.h"
#include <windows.h>
#include <fstream>


using namespace cv;
using namespace std;

#include "../OCR_Api.h"



Mat PreProcess(Mat& inImg)
{
    Mat imggray;
    if (inImg.channels() == 3)
    {
        cvtColor(inImg, imggray, CV_BGR2GRAY);
    }
    else
    {
        imggray = inImg;
    }

    blur(imggray, imggray, Size(3, 3));

    return imggray;
}

bool less_x(const Rect &rc1, const Rect &rc2)
{
    return rc1.x < rc2.x;
}

bool Segmentation(Mat& inImggray, Rect& outROI, vector<Rect>& outRects)
{
    vector<Mat> vecMat;
    int w = inImggray.cols;
    int h = inImggray.rows;

    Rect rcROI(0.05*w, 0.2*h, 0.9*w, 0.6*h);
    outROI = rcROI;
    Mat imgROI = inImggray(rcROI).clone();

    Mat imgROIBin;
    threshold(imgROI, imgROIBin, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY_INV);

    Mat elem;
    elem = getStructuringElement(MORPH_RECT, Size(1, 3));
    for (int i = 0; i < 6; i++)
    {
        morphologyEx(imgROIBin, imgROIBin, MORPH_DILATE, elem);
    }

#ifdef NO_USE
    vector< vector<Point> > contours;
    findContours(imgROIBin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    vector< vector<Point> >::iterator itc = contours.begin();
    vector<RotatedRect> outRects;

    while (itc != contours.end())
    {
        RotatedRect mr = minAreaRect(Mat(*itc));
        ++itc;
        outRects.push_back(mr);
    }


    Point2f vertices[4];
    vector<RotatedRect>::iterator iter;
    cout << "chars:" << endl;
    for (iter = outRects.begin(); iter != outRects.end(); iter++)
    {
        (*iter).points(vertices);
        Rect rc = (*iter).boundingRect();

        if (rc.width > 0.05*w && rc.height > 0.09*h)
        {
            Mat roi = imgROI(rc).clone();
            vecMat.push_back(roi);
        }

    }
#endif

    vector< vector<Point> > contours;
    cv::findContours(imgROIBin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    vector< vector<Point> >::iterator itc = contours.begin();

    outRects.clear();
    while (itc != contours.end())
    {
        Rect rc = boundingRect(*itc);
        ++itc;
        if (rc.width > 0.05*w && rc.height > 0.09*h)
        {
            outRects.push_back(rc);
        }
    }

    sort(outRects.begin(), outRects.end(), less_x);

    return true;
}


int PredictChar0(Mat& inChar)
{
    CvSVM SVM;
    SVM.load("mnist_svm.xml");
    Mat img0;
    inChar.convertTo(img0, CV_32FC1, 1 / 255.0);
    Mat img1;
    img1 = img0.reshape(0, 1);
    return  SVM.predict(img1);
}


#define EACH_NUM_WIDTH       (9)
#define EACH_NUM_HEIGHT      (15)

int PredictChar(Mat& inChar)
{
    static int rd_cnt = 0;
    CvSVM SVM;

    //if (rd_cnt <= 0)
    {
        SVM.load("SVM_CODE_REC_MODEL.xml");
        rd_cnt++;
    }

    Mat inChar1;
    resize(inChar, inChar1, Size(EACH_NUM_WIDTH, EACH_NUM_HEIGHT), 0, 0, CV_INTER_LINEAR);

    Mat matFeature = inChar1.reshape(0, 1);
    matFeature.convertTo(matFeature, CV_32FC1);
    int num = (int)SVM.predict(matFeature);
    return num;
}

Mat GetSVMFeature(Mat inChar)
{
    const int hog_pic_w = 48;
    const int hog_pic_h = 48;
    HOGDescriptor hog(Size(hog_pic_w, hog_pic_h), Size(16, 16), Size(8, 8), Size(8, 8), 9); // 48x48  900

    Mat matCharGray;
    if (inChar.channels() == 3)
    {
        cvtColor(inChar, matCharGray, CV_BGR2GRAY);
    }
    else
    {
        matCharGray = inChar;
    }

    Mat imggray1;
    resize(matCharGray, imggray1, Size(48, 48), 0, 0, CV_INTER_LINEAR);
    vector<float> vecDescriptors;//结果数组
    hog.compute(imggray1, vecDescriptors);

    Mat matFeatureRow(vecDescriptors);
    matFeatureRow = matFeatureRow.reshape(0, 1);

    Mat matOut = matFeatureRow.clone();
    return matOut;
}




/*****************************************************************************
*                       OcrInit
*  函 数 名： OcrInit
*  说    明：
*  参    数：handle:句柄
*            path1:路径1
*            path2:路径2
*  返 回 值：0 ： 成功，非零 ： 失败
*****************************************************************************/
APP_TEST_API int32_t __stdcall OcrInit(const void** handle, const char* path1, const char* path2)
{
    CvSVM *pSVM = NULL;
    pSVM = new CvSVM();
    pSVM->load(path1);
    if (pSVM->get_support_vector_count() <= 0)
    {
        return 1;
    }

    if (pSVM->get_support_vector_count() > 0)
    {
        *handle = (void*)pSVM;
        return 0;
    }
}



/*****************************************************************************
*                       OcrRelease
*  函 数 名： OcrRelease
*  说    明：
*  参    数：handle:句柄
*  返 回 值：0 ： 成功，非零 ： 失败
*****************************************************************************/
APP_TEST_API int32_t __stdcall OcrRelease(void* handle)
{
    if (handle == NULL)
    {
        return 3;
    }

    if (handle != NULL)
    {
        CvSVM *pTemp = (CvSVM *)handle;
        pTemp->clear();
        //delete pTemp;
        handle = NULL;
        return 0;
    }
}


/*****************************************************************************
*                       OCR
*  函 数 名： OcrRecognition
*  说    明：
*  参    数：PImageData:图像数据
*            pPRetStr:输出ocr=结果
*            ocr_type:ocr类型 
*                     英文数字混合 ocr_type=3000 任意长度英数混合，识别率会降低
*                     ocr_type=3010 1位英数
*                     ocr_type=3020 2位英数混合
*                     ocr_type=3100 10位英数混合
*  返 回 值：0 ： 成功，非零 ： 失败
*****************************************************************************/
APP_TEST_API int32_t __stdcall OcrRecognition(const void* handle, const ImageData * const PImageData, PRetStr pPRetStr, int ocr_type)
{
    /*if (!PImageData)
    {
        cout << "PImageData is null!" << endl;
        //return OCR_ERROR(OCR_ERROR_OVERFLOW);
        return 1;
    }

    if (!pPRetStr)
    {
        cout << "pPRetStr is null!" << endl;
        //return OCR_ERROR(OCR_ERROR_OVERFLOW);
        return 1;
    }

    if (PImageData->Buf == NULL)
   {
        cout << "PImageData->Buf is null!" << endl;
        //return OCR_ERROR(OCR_ERROR_OVERFLOW);
   }

    if (PImageData->Channels != 1 || PImageData->Channels != 3)
    {
        cout << "Channels is wrong!" << endl;
        //return OCR_ERROR(OCR_ERROR_CHANNEL);
    }

    if (PImageData->Width <= 0 || PImageData->Height <= 0)
    {
        cout << "Width or Height is wrong!" << endl;
        //return OCR_ERROR(OCR_ERROR_PARAMER);
    }*/

    if (handle == NULL)
    {
        return 3;
    }

    Mat img;
    if (ocr_type == 3040)
    {
        pPRetStr->RetCnt = 4;
        pPRetStr->RetBuf[0] = '1';
        pPRetStr->RetBuf[1] = '2';
        pPRetStr->RetBuf[2] = 'A';
        pPRetStr->RetBuf[3] = 'B';
        return 0;
    }


    if (ocr_type == 2000)
    {
        Mat matImg;

        CvSVM *pTemp = (CvSVM *)handle;

        if (3 == PImageData->Channels)
        {
            matImg.create(PImageData->Height, PImageData->Width, CV_8UC3);
            memcpy(matImg.data, PImageData->Buf, PImageData->Height * PImageData->Width * PImageData->Channels);
        }
        else
        {
            return 3;
        }

        Mat imgPro = PreProcess(matImg);
        Rect rcROI;
        vector<Rect> vecRects;
        bool bOk = Segmentation(imgPro, rcROI, vecRects);
        Mat matROI = matImg(rcROI);

        int iResults[10] = { -1 };

        for (int i = 0; i < vecRects.size(); i++)
        {
            Mat matChar = matROI(vecRects[i]);

            /*Mat matCharGray;
            if (matChar.channels() == 3)
            {
            cvtColor(matChar, matCharGray, CV_BGR2GRAY);
            }
            else
            {
            matCharGray = matChar;
            }

            Mat imggray1;
            resize(matCharGray, imggray1, Size(48, 48), 0, 0, CV_INTER_LINEAR);
            vector<float> vecDescriptors;//结果数组
            hog.compute(imggray1, vecDescriptors);

            Mat matFeatureRow(vecDescriptors);
            matFeatureRow = matFeatureRow.reshape(0, 1);*/
            Mat matFeatureRow = GetSVMFeature(matChar);
            int num = (int)pTemp->predict(matFeatureRow);
            iResults[i] = num;
        }

       
        if (vecRects.size() == 4 && iResults[2] == 12 && iResults[3] == 13)
        {
            int char13merge = vecRects[1].x - (vecRects[0].x + vecRects[0].width);
            if (char13merge >(vecRects[0].width + vecRects[1].width) / 4)
            {
                int W = vecRects[1].x + vecRects[1].width - vecRects[0].x;
                int x0 = vecRects[0].x;
                int y0 = vecRects[0].y;
                int h = max(vecRects[0].height, vecRects[1].height);


                Mat matChar0 = matROI(Rect(x0, y0, W / 3, h));
                Mat matFeature0 = GetSVMFeature(matChar0);
                iResults[0] = (int)pTemp->predict(matFeature0);


                Mat matChar1 = matROI(Rect(x0 + W / 3, y0, W / 3 - 1, h));
                Mat matFeature1 = GetSVMFeature(matChar1);
                iResults[1] = (int)pTemp->predict(matFeature1);

                Mat matChar2 = matROI(Rect(x0 + W * 2 / 3, y0, W / 3 - 1, h));
                Mat matFeature2 = GetSVMFeature(matChar2);
                iResults[2] = (int)pTemp->predict(matFeature2);
            }
        }


        int iRes = 0;
        char chop;

        if (vecRects.size() == 5 && iResults[3] == 12 && iResults[4] == 13 && iResults[0] <= 9 && iResults[2] <= 9 && (iResults[1] == 10 || iResults[1] == 11))
        {

            if (iResults[1] == 10)
            {
                chop = '+';
                iRes = iResults[0] + iResults[2];
            }

            if (iResults[1] == 11)
            {
                chop = 'X';
                iRes = iResults[0] * iResults[2];
            }
        }

        pPRetStr->RetCnt = 1;
        pPRetStr->RetBuf[0] = iRes;
        return 0;
    }
   
    return 1;
}