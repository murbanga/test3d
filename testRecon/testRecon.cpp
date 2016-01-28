#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_core247d.lib")
#pragma comment(lib, "opencv_highgui247d.lib")
#pragma comment(lib, "opencv_features2d247d.lib")
#pragma comment(lib, "opencv_nonfree247d.lib")
#pragma comment(lib, "opencv_calib3d247d.lib")
#pragma comment(lib, "opencv_imgproc247d.lib")
#else
#endif

using namespace std;
using namespace cv;

void reorder_keypoints(const vector<KeyPoint> keypoints[2],
		       const vector<DMatch> &matches,
		       vector<Point2f> points[2])
{
	points[0].resize(matches.size());
	points[1].resize(matches.size());
	for(size_t i = 0; i < matches.size(); ++i){
		DMatch m = matches[i];
		points[0][i] = keypoints[m.imgIdx][m.queryIdx].pt;
		points[1][i] = keypoints[m.imgIdx^1][m.trainIdx].pt;
	}
}

void detect_keypoints(Mat images[2], vector<Point2f> points[2], int detector_threshold, bool show_debug_windows)
{
	vector<KeyPoint> keypoints[2];
	Mat descr[2];

	SurfFeatureDetector detector(detector_threshold);
	SurfDescriptorExtractor extractor;

	for(int i = 0; i < 2; ++i){
		cout << i << " detect keypoints..." << endl;
		detector.detect(images[i], keypoints[i]);
		cout << "find keypoints " << keypoints[i].size() << endl;
		cout << i << " extract features..." << endl;
		extractor.compute(images[i], keypoints[i], descr[i]);

		if(show_debug_windows){
			Mat img_keypoints;
			drawKeypoints(images[i], keypoints[i], img_keypoints);
			imshow((i ? "Left keypoints" : "Right keypoints"), img_keypoints);
		}
	}

	cout << "find matches..." << endl;
	BFMatcher matcher(NORM_L2, true);
	vector<DMatch> matches;
	matcher.match(descr[0], descr[1], matches);

	if(show_debug_windows){
		Mat img_matches;
		drawMatches(images[0], keypoints[0], images[1], keypoints[1], matches, img_matches);
		namedWindow("Matches", CV_WINDOW_NORMAL);
		imshow("Matches", img_matches);
	}

	reorder_keypoints(keypoints, matches, points);
}

int main(int argc, char **argv)
{
	bool show_debug_windows = true;
	bool check_fundamental_matrix = true;
	int detector_threshold = 1200;
	string image_filename[2];

	for(int i = 0; i < argc; ++i){
		if(string(argv[i]) == "-l")image_filename[0] = argv[++i];
		else if(string(argv[i]) == "-r")image_filename[1] = argv[++i];
		else if(string(argv[i]) == "-v")show_debug_windows = true;
		else if(string(argv[i]) == "-check_fund")check_fundamental_matrix = true;
	}

	Mat images[2];

	for(int i = 0; i < 2; ++i){
		Mat image;
		image = imread(image_filename[i], CV_LOAD_IMAGE_COLOR);
		if(!image.data){
			cout << "fail to open image \"" << image_filename[i] << "\"" << endl;
			return -1;
		}
		cvtColor(image, images[i], COLOR_BGR2GRAY);
	}

	vector<Point2f> key_points[2];
	detect_keypoints(images, key_points, detector_threshold, show_debug_windows);

	cout << "find fundamental matrix..." << endl;
	vector<uchar> mask(key_points[0].size());
	Mat fund = findFundamentalMat(key_points[0], key_points[1], CV_FM_RANSAC, 4.0, 0.99, mask);
	cout << fund << endl;

	if(check_fundamental_matrix){
		for(int i = 0; i < 2; ++i){
			vector<Point3f> lines;
			float error = 0;
			const vector<Point2f> &versa = key_points[i^1];

			computeCorrespondEpilines(key_points[i], i + 1, fund, lines);

			for(size_t j = 0; j < versa.size(); ++j){
				error += abs(versa[j].x*lines[j].x + versa[j].y*lines[j].y + lines[j].z);
			}
			cout << i << " error total " << error << " mean " << error/versa.size() << endl;
		}
	}

	waitKey(0);

	return 0;
}