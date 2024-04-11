#ifndef STATEYE2_VIDEOTRACKER_H
#define STATEYE2_VIDEOTRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/tracking.hpp>
#include <string>
#include <vector>

class VideoTracker {
public:
    VideoTracker(const std::string& videoPath, const std::string& cfgPath, const std::string& weightsPath,
                 const cv::Scalar& selectedColorHSV, const std::vector<cv::Point>& selectedPoints,
                 float baseScaleFactorWidth, float baseScaleFactorLengthNear, float baseScaleFactorLengthFar,
                 float baseScaleFactorGoal, float baseScaleFactorPenaltyBox);

    void initialiseTrackersWithDetections();
    void trackAndCalculateDistances();

    void run();

private:
    cv::VideoCapture video;
    cv::dnn::Net net;
    std::vector<cv::Ptr<cv::Tracker>> trackers;
    std::vector<cv::Point2f> previousPositions;
    std::vector<float> distancesTraveled;
    cv::Mat frame;

    cv::Scalar selectedColorHSV;
    std::vector<cv::Point> selectedPoints;

    float distanceHalfwayToCorner = 0.0f;
    float distanceHalfwayAcross = 0.0f;
    float distancePenaltyBox = 0.0f;

    float baseScaleFactorWidth;
    float baseScaleFactorLengthNear;
    float baseScaleFactorLengthFar;
    float baseScaleFactorGoal;
    float baseScaleFactorPenaltyBox;

    float calculateScaleFactorForPosition(float y, const cv::Point& nearHalfwayLine, const cv::Point& farHalfwayLine);
    bool hasSufficientColor(const cv::Mat& frame, const cv::Rect2d& bbox, const cv::Scalar& selectedColorHSV);
    void drawTrackingInfo(cv::Mat& frame, const cv::Rect2d& bbox, float distance);
};

#endif //STATEYE2_VIDEOTRACKER_H
