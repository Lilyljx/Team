#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <io.h> //查找文件相关函数
using namespace std;
using namespace cv;
using namespace ml;
ostringstream oss;//结合字符串和数字
int num = -1;
Mat dealimage;
Mat src;
Mat yangben_gray;
Mat yangben_thresh;
void feature(Mat dealimage); //读取训练数据
const int classsum = 1;//训练图片共有10类，可修改
const int imagesSum = 21;//每类有500张图片，可修改
int yangben_data_position = -1;
Mat data_mat = Mat(classsum*imagesSum, 8100, CV_32FC1);//data_mat用来保存所有训练样本的HOG特征，每一列为一副图像的HOG
//每一行为每一幅训练图像,必须为CV_32FC1的类型,行、列、类型；第二个参数，即矩阵的列是由下面的descriptors的大小决定的，
//可以由descriptors.size()得到，且对于不同大小的输入训练图片，这个值是不同的
//descriptors为提取到的HOG特征
int main()
{  	//训练数据，每一行一个训练图片
	Mat trainingData;	//训练样本标签
	Mat labels;	//最终的训练样本标签
	Mat clas;	//最终的训练数据
	Mat traindata; 	
	//从指定文件夹下提取图片


	for (int p = 0; p < classsum; p++)//依次提取0到9文件夹中的图片
	{		
		oss << "number";
		//num += 1;//num从0到9

		num = num + 3;
		int label = num;
		oss << num << "/*.png";//图片名字后缀，oss可以结合数字与字符串
		string pattern = oss.str();//oss.str()输出oss字符串，并且赋给pattern
		oss.str("");//每次循环后把oss字符串清空
		vector<Mat> input_images;
		vector<String> input_images_name;
		glob(pattern, input_images_name, false);//为false时，仅仅遍历指定文件夹内符合模式的文件
												//当为true时，会同时遍历指定文件夹的子文件夹	
												//此时input_images_name存放符合条件的图片地址

		int all_num = input_images_name.size();		//文件下总共有几个图片
		//cout << num << ":总共有" << all_num << "个图片待测试" << endl;
		for (int i = 0; i < imagesSum; i++)//依次循环遍历每个文件夹中的图片
		{
			cvtColor(imread(input_images_name[i]), yangben_gray, COLOR_BGR2GRAY);//灰度变换
			threshold(yangben_gray, yangben_thresh, 0, 255, THRESH_OTSU);//二值化
			//循环读取每张图片并且依次放在vector<Mat> input_images内
			input_images.push_back(yangben_thresh);
			dealimage = input_images[i]; 			//选择了HOG的方式完成特征提取工作

			yangben_data_position += 1;//代表为第几幅图像

			feature(dealimage);//图片特征提取
			labels.push_back(label);//把每个图片对应的标签依次存入
			cout << "第" << yangben_data_position << "样本正在提取HOG特征" << endl;
		}
	}
	cout << "样本特征提取完毕,等待创建SVM模型" << endl;

	//创建SVM模型
	// 创建分类器并设置参数
	Ptr<SVM> SVM_params = SVM::create();
	SVM_params->setType(SVM::C_SVC);//C_SVC用于分类，C_SVR用于回归
	SVM_params->setKernel(SVM::LINEAR);  //LINEAR线性核函数。SIGMOID为高斯核函数
	SVM_params->setDegree(0);//核函数中的参数degree,针对多项式核函数
	SVM_params->setGamma(1);//核函数中的参数gamma,针对多项式/RBF/SIGMOID核函数
	SVM_params->setCoef0(0);//核函数中的参数,针对多项式/SIGMOID核函数
	SVM_params->setC(1);//SVM最优问题参数，设置C-SVC，EPS_SVR和NU_SVR的参数
	SVM_params->setNu(0);//SVM最优问题参数，设置NU_SVC， ONE_CLASS 和NU_SVR的参数
	SVM_params->setP(0);//SVM最优问题参数，设置EPS_SVR 中损失函数p的值
						//结束条件，即训练1000次或者误差小于0.01结束
	SVM_params->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01));//训练数据和标签的结合
	Ptr<TrainData> tData = TrainData::create(data_mat, ROW_SAMPLE, labels); 	// 训练分类器
	SVM_params->train(tData);//训练 	//保存模型
	SVM_params->save("svm.xml");
	cout << "训练好了！！！" << endl;
	cout << "等待识别" << endl;
	
	//预测部分
	Mat src = imread("2.png");
	cvtColor(src, src, COLOR_BGR2GRAY);
	threshold(src, src, 0, 255, CV_THRESH_OTSU);
	imshow("原图像", src); 	//输入图像取特征点
	Mat trainTempImg = Mat::zeros(Size(128, 128), CV_8UC1);
	resize(src, trainTempImg, trainTempImg.size());
	HOGDescriptor *hog = new HOGDescriptor(Size(128, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	vector<float>descriptors;//结果数组
	hog->compute(trainTempImg, descriptors, Size(1, 1), Size(0, 0));
	//cout << "HOG描述子向量维数    " << descriptors.size() << endl;
	Mat SVMtrainMat = Mat(1, descriptors.size(), CV_32FC1);
	int  number1 = descriptors.size();	//将计算好的HOG描述子复制到样本特征矩阵SVMtrainMat
	for (int i = 0; i < number1; i++)
	{//把一幅图像的HOG描述子向量依次存入data_mat矩阵的同一列
	 //因为输入图像只有一个，即SVMtrainMat只有一列，则为0
		SVMtrainMat.at<float>(0, i) = descriptors[i];//  n++;
	}
	SVMtrainMat.convertTo(SVMtrainMat, CV_32FC1);//更改图片数据的类型，必要，不然会出错	
	int ret = (int)SVM_params->predict(SVMtrainMat);//检测结果
	cout << "识别的数字为：" << ret << endl;
	waitKey(0);
	return 0;
}
void feature(Mat dealimage)
{	//把训练样本放大到128，128。便于HOG提取特征
	Mat trainImg = Mat(Size(128, 128), CV_8UC1);
	resize(dealimage, trainImg, trainImg.size());
	//处理HOG特征
	//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9，需要修改
	HOGDescriptor *hog = new HOGDescriptor(Size(128, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	vector<float>descriptors;//存放结果    为HOG描述子向量
	hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //Hog特征计算，检测窗口移动步长(1,1)
    //cout << "HOG描述子向量维数    : " << descriptors.size() << endl;
	for (vector<float>::size_type j = 0; j < descriptors.size(); j++)
	{//把一幅图像的HOG描述子向量依次存入data_mat矩阵的同一列
		data_mat.at<float>(yangben_data_position, j) = descriptors[j];
	}
}
