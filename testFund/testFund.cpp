#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_core247d.lib")
#pragma comment(lib, "opencv_calib3d247d.lib")
#else
#pragma comment(lib, "opencv_core247.lib")
#pragma comment(lib, "opencv_calib3d247.lib")
#endif

using namespace std;
using namespace cv;

const int N = 7;
const float CELL_WIDTH = 0.1f;
const int POINTS_FOR_SEARCH = 16;

struct Cam
{
	Mat camera;
	Vec3f rvec;
	Vec3f tvec;
};

vector<Point3f> generateObject()
{
	vector<Point3f> pt(N*N);
	for(int i = 0, idx = 0; i < N; ++i)for(int j = 0; j < N; ++j, ++idx){
		pt[idx] = Point3f(i*CELL_WIDTH, j*CELL_WIDTH, 0);
	}
	return pt;
}

Vec3f normalize(const Vec3f &v)
{
	float len = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
	return v / len;
}

Vec3f cross(const Vec3f &a, const Vec3f &b)
{
	return Vec3f(
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0]);
}

Cam generateCam(float f, float width, float height, const Vec3f &eye, const Vec3f &center, const Vec3f &up)
{
	Cam cam;
	Mat camera = Mat::zeros(3,3, CV_32F);

	camera.at<float>(0,0) = f;
	camera.at<float>(1,1) = f;
	camera.at<float>(2,2) = 1;
	camera.at<float>(0,2) = width;
	camera.at<float>(1,2) = height;
	cam.camera = camera;

	// stole this code from gluLookAt function
	Vec3f forward = center - eye;
	forward = normalize(forward);

	Vec3f side = normalize(cross(forward, up));
	
	Vec3f up2 = cross(side, forward);

	Mat m(3,3, CV_32F);
	m.at<float>(0,0) = side[0];
	m.at<float>(0,1) = side[1];
	m.at<float>(0,2) = side[2];
	m.at<float>(1,0) = up2[0];
	m.at<float>(1,1) = up2[1];
	m.at<float>(1,2) = up2[2];
	m.at<float>(2,0) = -forward[0];
	m.at<float>(2,1) = -forward[1];
	m.at<float>(2,2) = -forward[2];
	Rodrigues(m, cam.rvec);

	cam.tvec = -eye;

	return cam;
}

void dump(const char *filename, const vector<Point2f> &vec)
{
	ofstream left_file(filename);
	if(left_file.is_open()){
		for(int i = 0; i < vec.size(); ++i)
			left_file << vec[i].x << " " << vec[i].y << endl;
	}
}

int main(int argc, char **argv)
{
	vector<Point3f> obj = generateObject();
	Mat fund;
	
	Cam left = generateCam(1.5, 1, 1, Vec3f(-1, -1, 5), Vec3f(0, 0, 0), Vec3f(1,0,0));
	cout << "left cam\n" << left.camera << endl;
	cout << "rvec\n" << left.rvec << endl;
	cout << "tvec\n" << left.tvec << endl;

	Cam right = generateCam(1, 1, 1, Vec3f(0, 1, 5), Vec3f(0, 0, 0), Vec3f(1,0,0));
	cout << "right cam\n" << right.camera << endl;
	cout << "rvec\n" << right.rvec << endl;
	cout << "tvec\n" << right.tvec << endl;

	vector<Point2f> left_picture;
	vector<Point2f> right_picture;

	vector<Point3f> left_lines;
	vector<Point3f> right_lines;

	try{
		projectPoints(obj, left.rvec, left.tvec, left.camera, noArray(), left_picture);
		dump("left_proj.txt", left_picture);
		projectPoints(obj, right.rvec, right.tvec, right.camera, noArray(), right_picture);
		dump("right_proj.txt", right_picture);

#if 0
		vector<Point2f> partial_left_picture(POINTS_FOR_SEARCH);
		vector<Point2f> partial_right_picture(POINTS_FOR_SEARCH);

		for(int i = 0; i < POINTS_FOR_SEARCH; ++i){
			partial_left_picture[i] = left_picture[3*i];
			partial_right_picture[i] = right_picture[3*i];
		}

		fund = findFundamentalMat(partial_left_picture, partial_right_picture, CV_FM_8POINT);
#else
		// leonid.volnin: for some reason RANSAC returns zero fundamental matrix, if I pass not all points (see below)
		fund = findFundamentalMat(left_picture, right_picture, CV_FM_RANSAC);
#endif
		cout << "fundamental matrix\n" << fund << endl;

		computeCorrespondEpilines(left_picture, 1, fund, right_lines);
		computeCorrespondEpilines(right_picture, 2, fund, left_lines);
	}catch(cv::Exception e){
		cout << e.what();
	}
	
	float left_error = 0;
	float right_error = 0;
	for(int i = 0; i < obj.size(); ++i){
		Point3f line = left_lines[i];
		Point2f point = left_picture[i];
		
		float error = abs(line.x*point.x + line.y*point.y + line.z);
		left_error += error;

		line = right_lines[i];
		point = right_picture[i];
		error = abs(line.x*point.x + line.y*point.y + line.z);
		right_error += error;
	}
	cout << "left error " << left_error << endl << "right error " << right_error << endl;
	return 0;
}