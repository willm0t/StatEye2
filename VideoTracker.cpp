#include "VideoTracker.h"

VideoTracker::VideoTracker(const std::string &videoPath, const std::string &cfgPath, const std::string &weightsPath,
                           const cv::Scalar& selectedColorHSV, const std::vector<cv::Point>& selectedPoints,
                           float baseScaleFactorWidth, float baseScaleFactorLengthNear, float baseScaleFactorLengthFar,
                           float baseScaleFactorGoal, float baseScaleFactorPenaltyBox)
        : video(videoPath), net(cv::dnn::readNetFromDarknet(cfgPath, weightsPath)),
          selectedColorHSV(selectedColorHSV), selectedPoints(selectedPoints),
          baseScaleFactorWidth(baseScaleFactorWidth), baseScaleFactorLengthNear(baseScaleFactorLengthNear),
          baseScaleFactorLengthFar(baseScaleFactorLengthFar), baseScaleFactorGoal(baseScaleFactorGoal),
          baseScaleFactorPenaltyBox(baseScaleFactorPenaltyBox) {

    net = cv::dnn::readNetFromDarknet(cfgPath, weightsPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    if (!video.isOpened()) {
        throw std::runtime_error("Failed to open video");
    }
}

void VideoTracker::initialiseTrackersWithDetections() {
    cv::Mat frame;
    if (!video.read(frame)) {
        throw std::runtime_error("Failed to read first frame from video.");
    }

    // Prepare the frame for YOLO (convert to blob)
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    // Forward pass to get detections
    std::vector<cv::Mat> detections;
    net.forward(detections, net.getUnconnectedOutLayersNames());

    // Extract bounding boxes and confidences
    std::vector<cv::Rect2d> bboxes;
    float confidenceThreshold = 0.5; // Threshold to filter detections
    for (auto &detection : detections) {
        for (int i = 0; i < detection.rows; i++) {
            float* data = detection.ptr<float>(i);
            float confidence = data[4];
            if (confidence > confidenceThreshold) {
                int centerX = static_cast<int>(data[0] * frame.cols);
                int centerY = static_cast<int>(data[1] * frame.rows);
                int width = static_cast<int>(data[2] * frame.cols);
                int height = static_cast<int>(data[3] * frame.rows);
                cv::Rect2d bbox(centerX - width / 2, centerY - height / 2, width, height);
                bboxes.push_back(bbox);
            }
        }
    }

        // Filter detections based on the selected color
    std::vector<cv::Rect2d> filteredBboxes;
    for (const auto &bbox : bboxes) {
        if (hasSufficientColor(frame, bbox, selectedColorHSV)) {
            filteredBboxes.push_back(bbox);
        }
    }

    // Initialise trackers for filtered detections
    for (const auto &bbox : filteredBboxes) {
        auto tracker = cv::TrackerCSRT::create();
        tracker->init(frame, bbox);
        trackers.push_back(tracker);
    }
}

bool VideoTracker::hasSufficientColor(const cv::Mat& frame, const cv::Rect2d& bbox, const cv::Scalar& colorHSV) {
    cv::Mat roi = frame(bbox);
    cv::Mat hsvRoi;
    cv::cvtColor(roi, hsvRoi, cv::COLOR_BGR2HSV);
    cv::Scalar lowerBound = selectedColorHSV - cv::Scalar(10, 50, 50); // Broaden the range
    cv::Scalar upperBound = selectedColorHSV + cv::Scalar(10, 255, 255);
    cv::Mat mask;
    cv::inRange(hsvRoi, lowerBound, upperBound, mask);
    double percentage = cv::countNonZero(mask) / (double)(roi.rows * roi.cols);

    std::cout << "Color Match Percentage: " << percentage << std::endl; // Debugging print

    double minPercentage = 0.05;
    return percentage >= minPercentage;
}

void VideoTracker::trackAndCalculateDistances() {
    while (video.read(frame)) { // Use the class member `video` and `frame`
        for (size_t i = 0; i < trackers.size(); ++i) {
            cv::Rect bbox;
            if (trackers[i]->update(frame, bbox)) {
                cv::Point2f currentPosition(bbox.x + bbox.width / 2.0, bbox.y + bbox.height);

                if (i >= previousPositions.size()) {
                    previousPositions.push_back(currentPosition);
                    distancesTraveled.push_back(0.0f);  // Reset the distance when tracker is newly initialized
                }

                double pixelsMoved = cv::norm(currentPosition - previousPositions[i]);
                if (pixelsMoved > 500) {  // Example threshold, adjust based on expected maximum movement
                    std::cout << "Warning: Possible tracker jump detected for tracker " << i << std::endl;
                    continue;  // Skip this frame for this tracker
                }

                float adjustedScaleFactor = calculateScaleFactorForPosition(currentPosition.y, selectedPoints[2], selectedPoints[3]);
                if (adjustedScaleFactor < 0) {  // Check for invalid scale factor
                    std::cerr << "Error: Negative scale factor computed." << std::endl;
                    continue;  // Skip this frame for this tracker
                }

                double realWorldDistanceMoved = pixelsMoved * adjustedScaleFactor;
                distancesTraveled[i] += realWorldDistanceMoved;

                drawTrackingInfo(frame, bbox, distancesTraveled[i]);
            }
        }

        cv::imshow("Tracking", frame);
        int key = cv::waitKey(30);
        if (key == 27) break; // ESC key to exit
    }
}


void VideoTracker::drawTrackingInfo(cv::Mat &frame, const cv::Rect2d &bbox, float distance) {
    cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2, 1);
    std::string distanceText = std::to_string(static_cast<int>(distance)) + " m";
    cv::Point textPosition = cv::Point(bbox.x, bbox.y - 10);
    cv::putText(frame, distanceText, textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
}

float VideoTracker::calculateScaleFactorForPosition(float y, const cv::Point& nearHalfwayLine, const cv::Point& farHalfwayLine) {
    float deltaY = std::abs(farHalfwayLine.y - nearHalfwayLine.y);
    if(deltaY == 0) deltaY = 1; // Prevent division by zero
    float scaleGradient = (baseScaleFactorLengthFar - baseScaleFactorLengthNear) / deltaY;

    float result = baseScaleFactorLengthNear + (y - nearHalfwayLine.y) * scaleGradient;

    // Debug output
    std::cout << "deltaY: " << deltaY << ", scaleGradient: " << scaleGradient << ", y: " << y
              << ", nearHalfwayLine.y: " << nearHalfwayLine.y << ", result: " << result << std::endl;

    return result;
}




