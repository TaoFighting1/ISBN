#include<opencv2/opencv.hpp>
#include<vector>
#include<iostream>
#include<cstring>
#include<map>
#include<cstring>
#include<algorithm>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>
#include<cmath>
#include <opencv2/highgui/highgui_c.h>
#include<iomanip>
using namespace cv;
using namespace std;
typedef pair<int, int> range;
const int INF = 0x3f3f3f3f;

//const string train_path = "C:/Users/JYT/Desktop/3ѵ���õ�ISBN��";
const string train_path = "2ѵ���õ�ISBN��";
//const string train_path = "C:/Users/JYT/source/repos/Secondary Projects/Secondary Projects/ѵ���õ�ISBN��/ISBN 978-7-112-14830-1.png";
const string model_path = "����";
const int N = 20;
vector<String> train_Imgname;//����ѵ��ͼƬ��
vector<String> model_Imgname;//����ģ����
vector<pair<string, Mat>> templs;//����ģ�������
int rtNums = 0, accNums = 0, sumNums = 0;//�ֱ𴢴���ȷ����׼ȷ������������
char modelchar[N], qiuchar[N];//�洢�ַ�����
int f[N][N];//��̬�滮�������������

//����ѵ��ͼƬ����
void dealString(string& str) {
	int t = str.find("ѵ��");
	string s = str.substr(t + 14);
	string str_result = "";
	for (int i = 0; i < s.size(); i++) {
		if (((s[i] - '0') >= 0 && (s[i] - '0') <= 9) || s[i] == 'X') {
			str_result += s[i];
		}
	}
	str = str_result;
}

//���������и�
Mat crosswisePartition2(Mat srcImage, bool flag) {
	Mat substrImage;
	int row = srcImage.rows, col = srcImage.cols;
	int rStart = -1, rEnd = -1;

	for (int i = 0; i < row; i++) {
		int sum_num = 0;
		for (int j = 0; j < col; j++) {
			if ((int)srcImage.at<uchar>(i, j) != 0) sum_num++;
		}
		if (sum_num > 0) {
			rStart = i;
			break;
		}
	}
	for (int i = row - 1; i > rStart && rStart != -1; i--) {
		int sum_num = 0;
		for (int j = 0; j < col; j++) {
			if ((int)srcImage.at<uchar>(i, j) != 0) sum_num++;
		}
		if (sum_num > 0) {
			rEnd = i + 1;
			break;
		}
	}
	if (rEnd > row) rEnd = row;

	if (rEnd > rStart + srcImage.rows / 3 && rStart != -1) {
		substrImage = Mat(srcImage, Range(rStart, rEnd), Range(0, col));
	}

	return substrImage;

}
int number = 0;
//�����и�
Mat crosswisePartition(Mat srcImage, int limit, int Nu, bool has) {//limit��������
	Mat substrImage;
	int row = srcImage.rows, col = srcImage.cols;
	int rStart = -1, rEnd = -1;
	bool has_first = false;
	for (int i = 0; i < row; i++) {
		int sum_num = 0;
		for (int j = 0; j < col; j++) {
			if ((int)srcImage.at<uchar>(i, j) != 0) sum_num++;
		}
		if (sum_num > Nu && !has_first) {
			rStart = i;
			has_first = true;
		}
		else if (sum_num <= Nu && has_first) {
			rEnd = i;
			has_first = false;
			if (rEnd - rStart < 5) {
				continue;
			}
			else {
				if (rEnd < limit) {
					has_first = true;
					i += limit;
				}
				else break;
			}
		}
	}
	if (has_first && rEnd == -1) rEnd = row;
	if (rEnd - rStart > 5) {
		substrImage = Mat(srcImage, Range(rStart, rEnd), Range(0, col));

	}
	return substrImage;
}
int num = 0;
//�����и�
vector<Mat> lengthways(Mat srcImage) {
	int row = srcImage.rows, col = srcImage.cols;
	vector<range> saveRange;
	vector<Mat> saveMat;
	int cStart = -1, cEnd = -1;
	bool has_first = false;
	for (int i = 0; i < col; i++) {
		long long sum = 0;
		for (int j = 0; j < row; j++) {
			sum += (int)srcImage.at<uchar>(j, i);
		}
		if (sum > 200 && !has_first) {
			cStart = i;
			has_first = true;
		}
		else if (sum < 200 && has_first) {
			cEnd = i;
			saveRange.push_back({ cStart ,cEnd });
			cStart = i + 1;
			has_first = false;
		}
	}
	if (has_first) saveRange.push_back({ cStart ,col });

	for (auto item : saveRange) {
		if (item.second - item.first > 2) {
			Mat subImg = Mat(srcImage, Range(0, row), Range(item.first, item.second));
			saveMat.push_back(subImg);
			//imshow(to_string(num++), subImg);
		}
	}
	return saveMat;
}

//�и���ĸ
vector<Mat> Partition(Mat srcImage) {

	srcImage = crosswisePartition(srcImage, 0, 15, false);
	//imshow("�������", srcImage);
	return lengthways(srcImage);

}

//�ҶȻ�
Mat change_gray(Mat srcImage) {
	Mat output(srcImage.rows, srcImage.cols, CV_8UC1);
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			Vec3d pixel = srcImage.at<Vec3b>(i, j);
			double x = 0.114 * (int)pixel[0] + 0.587 * (int)pixel[1] + 0.299 * (int)pixel[2];
			output.at<uchar>(i, j) = 255-(int)x;
		}
	}
	return output;
}

void write_strength(Mat& src) {
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if ((int)src.at<uchar>(i, j) != 0) src.at<uchar>(i, j) = 255;
		}

	}
}

//��ɫ��ת
void Gray_change_color(Mat& srcImage) {
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			int t = (int)srcImage.at<uchar>(i, j);
			srcImage.at<uchar>(i, j) = 255 - t;
		}
	}
}

//���������Ƕ�
double Get_TurnTheta(Mat inputImg) {
	//���㴹ֱ������
	Mat yImg;
	Sobel(inputImg, yImg, -1, 0, 1, 5);
	//ֱ�߼��
	vector<Vec2f>lines;
	HoughLines(yImg, lines, 1, CV_PI / 180, 180);

	//������ת�Ƕ�
	float thetas = 0;
	for (int i = 0; i < lines.size(); i++) {
		float theta = lines[i][1];
		thetas += theta;
	}

	if (lines.size() == 0) {//δ��⵽ֱ��
		thetas = CV_PI / 2;
	}
	else {//��⵽ֱ�ߣ�ȡƽ��ֵ
		thetas /= lines.size();
	}
	return thetas;
}

//ˮ������
typedef pair<int, int> coordinate;
void water(Mat& srcImage, int n) {
	int cnt = 255 - n;//��nΪ255ʱ������ת���ڱߡ�
	int dx[9] = { -1,-1,-1,0,0,0,1,1,1 };
	int dy[9] = { -1,0,1,-1,0,1,-1,0,1 };
	int row = srcImage.rows;
	int col = srcImage.cols;
	queue<coordinate> qu;
	map<coordinate, int> mp;
	for (int i = 0; i < row; i++) {
		for (int j = 0; (((i == 0 || i == row - 1) && j < col)) || (i > 0 && i < row - 1 && j < col); j++) {
			if ((int)srcImage.at<uchar>(i, j) == cnt) {
				srcImage.at<uchar>(i, j) = 255 - cnt;
				coordinate A = { i,j };
				qu.push(A);
				mp[A] = 1;
				while (!qu.empty()) {
					auto t = qu.front();
					qu.pop();
					A = { t.first, t.second };
					mp[A] = 0;
					for (int k = 0; k < 9; k++) {
						int x = t.first + dx[k];
						int y = t.second + dy[k];
						if (x >= 0 && x < row && y >= 0 && y < col && (int)srcImage.at<uchar>(x, y) == cnt) {
							srcImage.at<uchar>(x, y) = 255 - cnt;
							A = { x, y };
							if (mp[A] == 0) {
								qu.push(A);
								mp[A] = 1;
							}
						}
					}
				}
			}
			if (i > 0 && i < row - 1 && j == 0) j = col - 2;
		}
	}
}

//��ֵ�˲�
void median_Filter(Mat in, Mat& out, int kernoSize)//kernoSize����Զ����������е�Ԫ�ؽ��д���
{
	Mat linkMan;//�м�ֵ�����ֵ��out
	linkMan = in;
	int rows = in.rows;
	int cols = in.cols;
	int halfK = kernoSize / 2;
	out = Mat::zeros(rows, cols, in.type());//��֤������������ͬһ�����͵�ͼ
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if ((i < halfK) || i >= rows - halfK || (j < halfK) || j >= cols - halfK)//�߽�
			{
				linkMan.at<uchar>(i, j) = in.at<uchar>(i, j);
			}
			else
			{
				vector<uchar>tool;
				for (int q = i - halfK; q <= i + halfK; q++)
				{
					for (int w = j - halfK; w <= j + halfK; w++)
					{
						tool.push_back(in.at<uchar>(q, w));
					}
				}
				sort(tool.begin(), tool.end());
				linkMan.at<uchar>(i, j) = tool[kernoSize * kernoSize / 2];
			}
		}
	}
	out = linkMan;
}


//ͳ�ƻҶ�ֱ��ͼ���õ���ֵ
int GetThreshold(Mat& src) {
	int row = src.rows;
	int col = src.cols;
	double* pixel = new double[256];//������¼��Ӧ����ֵ����
	double sum_pixel = row * col;
	for (int i = 0; i < 256; i++) pixel[i] = 0;//��ʼ��
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			pixel[(int)src.at<uchar>(i, j)]++;//��¼����
		}
	}
	for (int i = 0; i < 256; i++) pixel[i] /= sum_pixel;//pixel����δ��¼��Ӧ����ֵ����������ֵ�еĸ���
	double max_var = 0;//��󷽲�
	int max_var_position = 0;//��¼��󷽲�λ��
	for (int i = 0; i < 256; i++) {
		//left_pro��¼����i��߸���,right_proo��¼����i�ұ߸���
		//sum_left��¼��߻Ҷ���ֵ/ͼƬ��С, sum_right��¼��߻Ҷ���ֵ/ͼƬ��С
		double left_pro = 0, right_pro = 0, sum_left = 0, sum_right = 0;
		for (int j = 0; j < 256; j++) {
			if (j <= i) {
				left_pro += pixel[j];
				sum_left += j * pixel[j];
			}
			else {
				right_pro += pixel[j];
				sum_right += j * pixel[j];
			}
		}
		sum_right /= right_pro;//sum_right���ڱ�ʾ�ұ߻Ҷ�ƽ��ֵ
		sum_left /= left_pro;//sum_left���ڱ�ʾ��߻Ҷ�ƽ��ֵ
		double var = left_pro * right_pro * pow((sum_left - sum_right), 2);//����󷽲�
		//������󷽲λ��
		if (max_var < var) {
			max_var = var;
			max_var_position = i;
		}
	}
	return max_var_position;
}

// ͼ�����Ƴ̶Ⱥ���(ƽ��������� (MAE))
int Similarity(Mat img, const vector<pair<string, Mat>>& temp1s) {
	int min = INF;
	string rightValue = "";
	cv::resize(img, img, cv::Size(50, 40));
	for (const auto& templPair : temp1s) {
		const Mat& templ = templPair.second;
		int value = 0;
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				// ע�⣺�������img��templ���Ƕ�ֵͼ��
				value += (img.at<uchar>(i, j) - templ.at<uchar>(i, j)) * (img.at<uchar>(i, j) - templ.at<uchar>(i, j));
			}
		}
		if (value < min) {
			min = value;
			rightValue = templPair.first; // ����������ģ����ļ���
		}
	}//�����ISBNX-����-1;
	if (rightValue[0] == 'N') {
		return -1;
	}

	if (rightValue[0] == 'S' || rightValue[0] == 'B' || rightValue[0] == '-' || rightValue[0] == 'I') return -2;
	if (rightValue[0] == 'X') return -3;
	//cout <<"rightValue"<< rightValue << endl;
	return stoi(rightValue.substr(0, 1));
}// ����������ģ����ļ���(����)��

//������������У���̬�滮
int calculate(string model, string s) {
	int n1 = model.size(), n2 = s.size();
	for (int i = 1; i <= n1; i++) modelchar[i] = model[i - 1];
	for (int i = 1; i <= n2; i++) qiuchar[i] = s[i - 1];
	memset(f, 0, sizeof f);
	for (int i = 1; i <= n1; i++)
		for (int j = 1; j <= n2; j++)
		{
			f[i][j] = max(f[i - 1][j], f[i][j - 1]);
			if (modelchar[i] == qiuchar[j]) f[i][j] = max(f[i][j], f[i - 1][j - 1] + 1);
		}
	return f[n1][n2];
}

//ģ��ƥ��
string Template_matching(vector<Mat> srcVector) {
	string st = "";
	bool signal = false;
	for (int i = 0; i < srcVector.size(); i++) {

		//Mat	srcImg = crosswisePartition(srcVector[i], srcVector[i].rows/2,0,true);
		Mat	srcImg = crosswisePartition2(srcVector[i], true);
		//Mat	srcImg = srcVector[i];
		if (srcImg.rows > 0 && srcImg.cols > 0) {
			int t = Similarity(srcImg, templs);
			if ((t == -1) || ((t == 7 || t == 9) && !signal)) {
				st = "";
				signal = true;
				//cout << "��ʼ������" << endl;
			}
			if (signal) {
				if (t != -1 && t != -2 && t != -3) {
					st += to_string(t);
					//cout << "������" << endl;
				}
				else if (t == -3) {
					st += 'X';
					//cout << "�ӷ�����" << endl;
				}
			}
			//cout << t<<" "<<st << endl;
		}

	}
	return st;
}

//��ת�Ҷ�ͼ��
Mat Rotate(Mat srcImage) {
	Mat dst2 = srcImage;
	double thetas = Get_TurnTheta(dst2);
	thetas = 180 * thetas / CV_PI - 90;
	//��ת��ֵͼ��

	Mat M = getRotationMatrix2D(Point(srcImage.rows / 2, srcImage.cols / 2), thetas, 1);
	cv::warpAffine(dst2, srcImage, M, srcImage.size());
	return srcImage;
}

//��ʼ������ѵ��ͼƬ��ģ�岢����ͼƬ��
void init() {
	glob(train_path, train_Imgname, false);//��ȡѵ��ͼƬ����
	glob(model_path, model_Imgname, false);
	for (int i = 0; i < model_Imgname.size(); i++) {
		Mat src = imread(model_Imgname[i], IMREAD_GRAYSCALE);
		src = crosswisePartition2(src, false);
		string s = model_Imgname[i].substr(5, 3);
		cout << s << endl;
		cv::resize(src, src, cv::Size(50, 40));
		templs.push_back({ s, src });
	}
}

//��֤ISBN����ԭ��
bool verification(string str) {
	int n = str.size();
	if (n != 10 || n != 13) return false;
	int Num[13];
	for (int i = 0; i < n; i++) {
		Num[i] = str[i] - '0';
	}
	int sum = 0;
	if (n == 10) {
		for (int i = 0; i < 9; i++) {
			sum += Num[i] * (10 - i);
		}
		int result = 11 - sum % 11;
		if (result == 10 && str[9] == 'X') return true;
		if (result = Num[9]) return true;
		return false;
	}
	else {
		for (int i = 0; i < 12; i++) {
			if (i % 2 == 0) {
				sum += Num[i];
			}
			else {
				sum += 3 * Num[i];
			}
		}
		int result = 10 - sum % 10;
		if (result = Num[12]) return true;
		return false;
	}
}

int main() {
	init();
	int testImgNums = train_Imgname.size();
	for (int index = 0; index < testImgNums; index++) {
		int best_rate = 0;//��ǰ��õ���ȷʶ����������
		double best_right_char = 0;//��õ���ȷ�ַ�ʶ��
		int best_char_sum = 0;//�ַ�����
		Mat srcImage = imread(train_Imgname[index]);
		//����ģ������
		dealString(train_Imgname[index]);
		//������С
		Mat src;
		double width = 400;
		double height = width * srcImage.rows / srcImage.cols;
		resize(srcImage, srcImage, Size(width, height));
		//�ҶȻ�
		Mat gray, Gray;
		gray = change_gray(srcImage);
		Gray = gray;
		string model = train_Imgname[index];
		string best_str = "";
		bool flag = false;//��־�Ƿ������ǰ�˳�ѭ��
		for (int k = 1; k <= 2; k++) {//�����Ƿ����ˮ������
			/*���*/
			//for (int i = 3; i <= 9; i += 2) {//������ֵ�˲��ľ���˴�С
				for (double j = 0.8; j < 1.3; j += 0.05) {//��������������ֵ
					Mat inputsrc = gray;
					Mat ImgClear, OTSU;
					medianBlur(inputsrc, ImgClear, 3);//��ֵ�˲�
					//median_Filter(inputsrc, ImgClear, i);
					Gray_change_color(Gray);//��ת��ɫ
					int threshold_value = GetThreshold(Gray);//�������ֵ
					threshold_value = threshold_value * j;//������ֵ��С
					cv::threshold(Gray, OTSU, threshold_value, 255, THRESH_BINARY);//��ɫ��ǿ
					//imshow(to_string(k) + to_string(i) + to_string(j), OTSU);
					if (k == 1)water(OTSU, 255);
					//water(OTSU, 255);
					Gray_change_color(OTSU);//��ɫ��ת����ת�����ĺڱ߲�����Ҫ����ˮ������
					Mat  turnBin;
					turnBin = Rotate(OTSU);//ͼƬ��ת
					vector<Mat> saveMat = Partition(turnBin);//�ַ��и�
					waitKey(0);
					string s = Template_matching(saveMat);
					flag = verification(s);//�ж�ʶ�������ISBN���Ƿ���ϱ������
					if (flag && s != model) flag = false;//���Ϲ��ɵ�����ȾͲ��˳�
					int Subsequences = calculate(model, s);//��¼�����������
					double rate = Subsequences / (double)model.size();
					//if (rate >= best_rate && best_right_char <= Subsequences) {//ʶ���ʳ��ָ��ߵģ��͸��¼�¼ֵ
					if (rate >= best_rate && best_right_char <= Subsequences) {//ʶ���ʳ��ָ��ߵģ��͸��¼�¼ֵ
						best_right_char = Subsequences;//��¼ʶ��Ч����õ������ַ�����
						best_str = s;//��¼ʶ��Ч����õ��ַ���
						//cout << s << endl;
						best_rate = rate;
					}
					if ((best_right_char == model.size()&&best_right_char/(double)s.size()==1) || flag) {
						flag = true;
						break;
					}
				}
				if (flag) break;//����ƥ����˳�
			//}
			//if (flag) break;
		}
		if (flag) rtNums++;
		cout << "model:"<<model << endl;
		cout << "ʶ��: "<<best_str << endl;
		
		accNums += best_right_char;
		sumNums += model.size();
		cout.setf(ios::fixed);
		cout << setprecision(1) << "ʵʱ�ַ�׼ȷ��:" << (double)accNums/ sumNums*100 <<"%      "<< accNums << "/" << sumNums << endl;
		cout << "��ʶ����:" << (index + 1) << "  Ŀǰ��ȷ��:" << rtNums << endl;
		cout << endl;
	}
	cout.setf(ios::fixed);
	cout <<setprecision(2)<< "��ȷ��:" << (double)rtNums / testImgNums*100 << "%  ��ȷ��:" << (double)accNums / sumNums*100<<"%";
	system("pause");
	return 0;
}