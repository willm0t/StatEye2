#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include "Setup.h"
#include "VideoTracker.h"

int main() {
    try{
        std::string videoPath = "/Users/robertwillmot/Documents/University/Computer Science/3rd year/COMP6013 Dis/img_vid/clip3.mp4";
        std::string cfgPath = "/Users/robertwillmot/darknet/cfg/yolov4.cfg";
        std::string weightsPath = "/Users/robertwillmot/darknet/yolov4.weights";

        Setup setup(videoPath);
        setup.selectColourAndPoints();
        setup.handleDistanceInput();

        cv::Scalar selectedColourHSV = setup.getSelectedColourHSV();
        std::vector<cv::Point> selectedPoints = setup.getSelectedPoints();

        float baseScaleFactorWidth = setup.getBaseScaleFactorWidth();
        float baseScaleFactorLengthNear = setup.getBaseScaleFactorLengthNear();
        float baseScaleFactorLengthFar = setup.getBaseScaleFactorLengthFar();
        float baseScaleFactorGoal = setup.getBaseScaleFactorGoal();
        float baseScaleFactorPenaltyBox = setup.getBaseScaleFactorPenaltyBox();

        VideoTracker videoTracker(videoPath, cfgPath, weightsPath,
                                  selectedColourHSV, selectedPoints,
                                  baseScaleFactorWidth, baseScaleFactorLengthNear,
                                  baseScaleFactorLengthFar, baseScaleFactorGoal,
                                  baseScaleFactorPenaltyBox);

        videoTracker.initialiseTrackersWithDetections();
        videoTracker.trackAndCalculateDistances();
    }
    catch (const std::exception& e){
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

