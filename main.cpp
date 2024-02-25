#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>

cv::Scalar selectedColorHSV; // Global variable to store the selected color in HSV
void onMouse(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    cv::Mat* frame = reinterpret_cast<cv::Mat*>(userdata);
    cv::Vec3b bgrColor = frame->at<cv::Vec3b>(cv::Point(x, y));
    cv::Mat bgrMat(1, 1, CV_8UC3, bgrColor);

    cv::Mat hsvMat;
    cv::cvtColor(bgrMat, hsvMat, cv::COLOR_BGR2HSV);
    selectedColorHSV = hsvMat.at<cv::Vec3b>(0, 0);

    std::cout << "Selected Color (HSV): " << selectedColorHSV << std::endl;
}

bool hasSufficientColor(const cv::Mat& frame, const cv::Rect2d& bbox, const cv::Scalar& selectedColorHSV, double minPercentage = 0.05) {
    cv::Mat roi = frame(bbox);
    cv::Mat hsvRoi;
    cv::cvtColor(roi, hsvRoi, cv::COLOR_BGR2HSV);
    cv::Scalar lowerBound = selectedColorHSV - cv::Scalar(10, 50, 50); // Broaden the range
    cv::Scalar upperBound = selectedColorHSV + cv::Scalar(10, 255, 255);
    cv::Mat mask;
    cv::inRange(hsvRoi, lowerBound, upperBound, mask);
    double percentage = cv::countNonZero(mask) / (double)(roi.rows * roi.cols);

    std::cout << "Color Match Percentage: " << percentage << std::endl; // Debugging print

    return percentage >= minPercentage;
}


int main() {
    // Load YOLO network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet("/Users/robertwillmot/darknet/cfg/yolov4.cfg",
                                                   "/Users/robertwillmot/darknet/yolov4.weights");

    // Open the video file
    cv::VideoCapture video(
            "/Users/robertwillmot/Documents/University/Computer Science/3rd year/COMP6013 Dis/img_vid/clip1.mp4");
    if (!video.isOpened()) {
        std::cerr << "Failed to open video" << std::endl;
        return -1;
    }

    // Read the first frame
    cv::Mat frame;
    if (!video.read(frame)) {
        std::cerr << "Failed to read video" << std::endl;
        return -1;
    }

    cv::namedWindow("Select Color");
    cv::setMouseCallback("Select Color", onMouse, reinterpret_cast<void*>(&frame));

// Show the frame and wait for the user to select a color
    cv::imshow("Select Color", frame);
    cv::waitKey(0); // Wait indefinitely until a key is pressed or color is selected

    cv::destroyWindow("Select Color"); // Close the window

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
    for (auto &detection: detections) {
        for (int i = 0; i < detection.rows; i++) {
            float *data = detection.ptr<float>(i);
            float confidence = data[4];
            if (confidence > confidenceThreshold) {
                // Scale to frame size
                int centerX = (int) (data[0] * frame.cols);
                int centerY = (int) (data[1] * frame.rows);
                int width = (int) (data[2] * frame.cols);
                int height = (int) (data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                cv::Rect2d bbox(left, top, width, height);
                bboxes.push_back(bbox);
            }
        }
    }

    std::vector<cv::Rect2d> filteredBboxes;
    // Filter detections based on the selected color
    for (const auto& bbox : bboxes) {
        if (hasSufficientColor(frame, bbox, selectedColorHSV)) {
            filteredBboxes.push_back(bbox);
        }
    }

    // Initialize trackers for filtered detections
    std::vector<cv::Ptr<cv::Tracker>> trackers;
    for (const auto& bbox : filteredBboxes) {
        auto tracker = cv::TrackerCSRT::create();
        tracker->init(frame, bbox);
        trackers.push_back(tracker);
    }

// Process video and track objects
    while (video.read(frame)) {
        for (size_t i = 0; i < trackers.size(); i++) {
            // Create a temporary cv::Rect for the update method
            cv::Rect bbox;

            // Update the tracker with the current frame, use cv::Rect directly
            if (trackers[i]->update(frame, bbox)) {
                // Tracker update succeeded, bbox is already a cv::Rect, so draw it directly
                cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2, 1);
            } else {
                std::cerr << "Tracking failure detected for tracker " << i << std::endl;
                // Optional: Handle tracking failure
            }
        }

        // Display the frame with tracked objects
        cv::imshow("Tracking", frame);

        // Exit if ESC pressed
        if (cv::waitKey(1) == 27) break; // ESC key
    }

    cv::destroyAllWindows();
    return 0;
}