// eye_dll_test.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include "Includes.h"
#include <windows.h>
#include <fstream>


void Training(void);
void PredNewTrainingDataset(void);
void TestModel(void);



using namespace cv;
using namespace std;


#define LITTLE_ENDIAN (1)
#define BIG_ENDIAN (0)
/*return 1: little-endian, return 0: big-endian*/
int checkCPUendian()
{
    union
    {
        unsigned int a;
        unsigned char b;
    }c;
    c.a = 1;
    return (c.b == 1);
}



/*
void CodePreProcess(void)
{
    //VideoWrite();return 0;
    int total = 0, total1 = 0, total2 = 0;
    double heiht = 0;
    double width = 0;
    for (int i = 1; i <= 100; i++)
    {
        sprintf(str, "./code/%03d.jpg", i);
        img0 = imread(str, CV_LOAD_IMAGE_UNCHANGED);
        if (img0.empty())
        {
            cout << "image read error!" << endl;
            return;
        }
        img = img0.clone();
        cvtColor(img, imggray, CV_RGB2GRAY);
        imshow("code", img);

        threshold(imggray, imgedge, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY_INV);
        imshow("imgedge0", imgedge);
        Mat elem;
        elem = getStructuringElement(MORPH_RECT, Size(1, 3));
        for (int i = 0; i<14; i++)
        {
            morphologyEx(imgedge, imgedge, MORPH_DILATE, elem);
        }
        vector< vector<Point> > contours;
        findContours(imgedge, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
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
        int k = 1;
        total1 += outRects.size();
        total2++;
        for (iter = outRects.begin(), k = 1; iter != outRects.end(); iter++, k++)
        {
            (*iter).points(vertices);
            //for(int i = 0; i < 4;++i)
            //    line(img,vertices[i],vertices[(i+1)%4],Scalar(0,0,255));

            Rect rc = (*iter).boundingRect();
            //rectangle(img,rc,Scalar(0,0,255));
            sprintf(str1, "./num/%03d_%1d.jpg", i, k);
            Mat roi = imggray(rc);
            imwrite(str1, roi);
            total++;
            heiht += rc.height;
            width += rc.width;

        }

        imshow("imgedge", imgedge);
        imshow("img", img);

        //waitKey(0);
    }

    /*cout << "num=" << total << endl;
    cout << "num1=" << total1 << endl;
    cout << "num2=" << total2 << endl;
    cout << "h=" << heiht << endl;
    cout << "w=" << width << endl;
    cout << "avg: h=" << heiht/total << "---w=" << width/total << endl;
}
*/

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
        
        if (rc.width > 0.05*w && rc.height > 0.09*h )
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



void Test(void)
{
    char buf[256];
    int idx = 0;
    for (int i = 0; i <= 1000; i++)
    {
        //sprintf_s(buf, "E:/ml_study/captcha/dataset/test/%06d.jpg", i);
        sprintf_s(buf, "E:/ml_study/captcha/dataset/data2_calc/1001-2000/%06d.jpg", i + 1000);
        //sprintf_s(buf, "E:/ml_study/captcha/dataset/test1_20/%06d.jpg", i);
        idx = i;
        cout << buf << endl;
        Mat matImg = imread(buf, CV_LOAD_IMAGE_UNCHANGED);
        if (matImg.empty())
        {
            continue;
        }

        Mat imgPro = PreProcess(matImg);
        Rect rcROI;
        vector<Rect> vecRects;
        bool bOk = Segmentation(imgPro, rcROI, vecRects);
        Mat matROI = matImg(rcROI);
        for (int i = 0; i < vecRects.size(); i++)
        {
            Mat matChar = matROI(vecRects[i]);
            Mat sv;
            resize(matChar, sv, Size(matChar.cols * 4, matChar.rows * 4), 0, 0, CV_INTER_LINEAR);
            sprintf_s(buf, "E:/ml_study/captcha/dataset/data2_calc/1001-2000_out/%06d_%d.jpg", idx, i+1000);
            //sprintf_s(buf, "E:/ml_study/captcha/dataset/test1_20_out/%06d_%d.jpg", idx, i);
            imwrite(buf, sv);

            /*imshow("char", matChar);
            if (i==0 || i==2)
            {
                int num = PredictChar(matChar);
                cout << num << " ";
            }*/
            //waitKey(0);
        }
        cout << endl;
    }
}


int _tmain(int argc, _TCHAR* argv[])
{
    //Test();
    //Training();
    //PredNewTrainingDataset();
    //TestModel();
    Mat img = cv::imread("E:/ml_study/captcha/captcha_git/ml/000100.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat img1;
    //cv::imdecode(img.data, CV_LOAD_IMAGE_GRAYSCALE, &img1);
    if (img.empty())
    {
        cout << "img empty" << endl;
    }
    imshow("img", img);
    waitKey(0);
	return 0;
}





//////////////////////////////////////////////////////////////////////////
//training
//////////////////////////////////////////////////////////////////////////

#define CLASSES_NUM (14)
#define SAMPLE_NUM  (5)

void Training(void)
{
    CvSVM SVM;


    const int hog_pic_w = 48;
    const int hog_pic_h = 48;
    HOGDescriptor hog(Size(hog_pic_w, hog_pic_h), Size(16, 16), Size(8, 8), Size(8, 8), 9); // 48x48  900

    int k = hog.getDescriptorSize();
    cout << "k=" << k << endl;
    
    Mat matTrainingData(Size(k, 0), CV_32FC1);
    Mat matTrainingLabel(Size(1, 0), CV_32FC1);

    char buf[256];
    int idx = 0;
    for (int i = 0; i < CLASSES_NUM; i++)
    {
        for (int j = 0; j < SAMPLE_NUM; j++)
        {
            sprintf_s(buf, "E:/ml_study/captcha/training_dataset/type2_calc1/%d/%06d.jpg", i, j);
            idx = i;
            cout << buf << endl;
            Mat matImg = imread(buf, CV_LOAD_IMAGE_UNCHANGED);
            if (matImg.empty())
            {
                continue;
            }

            Mat matImgGray;
            if (matImg.channels() == 3)
            {
                cvtColor(matImg, matImgGray, CV_BGR2GRAY);
            }
            else
            {
                matImgGray = matImg;
            }

            Mat imggray1;
            resize(matImgGray, imggray1, Size(48, 48), 0, 0, CV_INTER_LINEAR);
            vector<float> vecDescriptors;//结果数组
            hog.compute(imggray1, vecDescriptors);
            
            /*
            vector<float> vecFeature;
            for (vector<float>::iterator it = descriptors.begin(); it != descriptors.end(); it++)
            {
                vecFeature.push_back(*it);
            }
            Mat matFeatureRow(vecFeature);
            matFeatureRow = matFeatureRow.reshape(0, 1);
            matTrainingData.push_back(matFeatureRow);
            */

            //resize(matImgGray, matImgGray, Size(50, 50), 0, 0, CV_INTER_LINEAR);
            //Mat matTmp;
            //matImgGray.convertTo(matTmp, CV_32FC1, 1 / 255.0);
            //Mat matTmp1 = matTmp.reshape(0, 1);
            
            Mat matFeatureRow(vecDescriptors);
            matFeatureRow = matFeatureRow.reshape(0, 1);
            matTrainingData.push_back(matFeatureRow);
            matTrainingLabel.push_back(float(i));
            
        }
    }


    cout << matTrainingData.rows << endl;
    cout << matTrainingLabel.rows << endl;

    CvSVMParams params;
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    SVM.train(matTrainingData, matTrainingLabel, Mat(), Mat(), params);
    SVM.save("SVM_type2_calc1.xml");

    int cnt = SVM.get_support_vector_count();
    cout << "cnt=" << cnt << endl;
    for (int i = 0; i < cnt; i++)
    {
        const float* v = SVM.get_support_vector(i);
        cout << *v << " ";
    }
    cout << endl << endl;
}


void PredNewTrainingDataset(void)
{
    CvSVM SVM;
    SVM.load("SVM_type2_calc0.xml");
    //SVM.load("SVM_CODE_REC_MODEL_OK.xml");


    if (SVM.get_support_vector_count() <= 0)
    {
        cout << "SVM is empty, please load xml model!" << endl;
        return;
    }


    const int hog_pic_w = 48;
    const int hog_pic_h = 48;
    HOGDescriptor hog(Size(hog_pic_w, hog_pic_h), Size(16, 16), Size(8, 8), Size(8, 8), 9); // 48x48  900

    char buf[256];
    int idx = 0;
    int pred_idx[CLASSES_NUM] = { 0 };

    for (int i = 0; i <= 10000; i++)
    {
        //sprintf_s(buf, "E:/ml_study/captcha/dataset/test/%06d.jpg", i);
        //sprintf_s(buf, "E:/ml_study/captcha/dataset/data2_calc/1001-2000/%06d.jpg", i + 1000);
        sprintf_s(buf, "E:/ml_study/captcha/dataset/data2_calc/yzm/%06d.jpg", i);

        //sprintf_s(buf, "E:/ml_study/captcha/dataset/test1_20/%06d.jpg", i);
        idx = i;
        cout << buf << endl;
        Mat matImg = imread(buf, CV_LOAD_IMAGE_UNCHANGED);
        if (matImg.empty())
        {
            continue;
        }

        Mat imgPro = PreProcess(matImg);
        Rect rcROI;
        vector<Rect> vecRects;
        bool bOk = Segmentation(imgPro, rcROI, vecRects);
        Mat matROI = matImg(rcROI);
        //Mat matROI = imgPro(rcROI);
        for (int i = 0; i < vecRects.size(); i++)
        {
            Mat matChar = matROI(vecRects[i]);
            Mat sv;
            //resize(matChar, sv, Size(matChar.cols * 4, matChar.rows * 4), 0, 0, CV_INTER_LINEAR);
            sv = matChar;
            
            Mat matCharGray;
            if (matChar.channels() == 3)
            {
                cvtColor(matChar, matCharGray, CV_BGR2GRAY);
            }
            else
            {
                matCharGray = matChar;
            }

            /*Mat matChar1;
            resize(matCharGray, matChar1, Size(50, 50), 0, 0, CV_INTER_LINEAR);

            Mat matFeature = matChar1.reshape(0, 1);
            matFeature.convertTo(matFeature, CV_32FC1);*/

            Mat imggray1;
            resize(matCharGray, imggray1, Size(48, 48), 0, 0, CV_INTER_LINEAR);
            vector<float> vecDescriptors;//结果数组
            hog.compute(imggray1, vecDescriptors);

            Mat matFeatureRow(vecDescriptors);
            matFeatureRow = matFeatureRow.reshape(0, 1);
            int num = (int)SVM.predict(matFeatureRow);

            sprintf_s(buf, "E:/ml_study/captcha/training_dataset/type2_calc1_pred/%d/%06d.jpg", num, pred_idx[num]++);
            imwrite(buf, matChar);
        }
    }
}



void TestModel(void)
{
    CvSVM SVM;
    SVM.load("SVM_type2_calc1.xml");
    if (SVM.get_support_vector_count() <= 0)
    {
        cout << "SVM is empty, please load xml model!" << endl;
        return;
    }

    const int hog_pic_w = 48;
    const int hog_pic_h = 48;
    HOGDescriptor hog(Size(hog_pic_w, hog_pic_h), Size(16, 16), Size(8, 8), Size(8, 8), 9); // 48x48  900

    char buf[256];
    int idx = 0;
    int pred_idx[CLASSES_NUM] = { 0 };

    ofstream ofresult("result.txt");

    for (int i = 0; i <= 1000; i++)
    {
        idx = i;
        sprintf_s(buf, "E:/ml_study/captcha/test_dataset/type2_calc/%06d.jpg", i); 
        cout << buf << endl;
        Mat matImg = imread(buf, CV_LOAD_IMAGE_UNCHANGED);
        if (matImg.empty())
        {
            continue;
        }

        Mat imgPro = PreProcess(matImg);
        Rect rcROI;
        vector<Rect> vecRects;
        bool bOk = Segmentation(imgPro, rcROI, vecRects);
        Mat matROI = matImg(rcROI);
        
        
        //if (vecRects.size() > 5)
        //{
        //    cout << vecRects.size() << endl;
        //    imshow("matROI", matROI);
        //    waitKey(0);
        //    //continue;
        //}

        int iResults[10] = {-1};

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
            int num = (int)SVM.predict(matFeatureRow);
            iResults[i] = num;
        }

        //imshow("matROI", matROI);
        for (size_t i = 0; i < 5; i++)
        {
            cout << iResults[i] << " ";
            ofresult << iResults[i] << " ";
        }
        cout << endl;
        ofresult << endl;
        //waitKey(0);

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
                iResults[0] = (int)SVM.predict(matFeature0);


                Mat matChar1 = matROI(Rect(x0 + W / 3, y0, W / 3 - 1, h));
                Mat matFeature1 = GetSVMFeature(matChar1);
                iResults[1] = (int)SVM.predict(matFeature1);

                Mat matChar2 = matROI(Rect(x0 + W * 2 / 3, y0, W / 3 - 1, h));
                Mat matFeature2 = GetSVMFeature(matChar2);
                iResults[2] = (int)SVM.predict(matFeature2);
            }
        }

        for (size_t i = 0; i < 5; i++)
        {
            ofresult << iResults[i] << " ";
        }
        ofresult << endl;
        
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
        sprintf_s(buf, "%06d-%d%c%d=%d.jpg", idx, iResults[0], chop, iResults[2], iRes);
        ofresult << buf << endl;

        sprintf_s(buf, "E:/ml_study/captcha/test_dataset/type2_calc_pred/%06d-%d%c%d=%d.jpg", idx, iResults[0], chop, iResults[2], iRes);
        imwrite(buf, matImg);
    }
}

