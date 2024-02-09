#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/tracking.hpp>

int main() {
    // Load YOLO network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet("/Users/robertwillmot/darknet/cfg/yolov4.cfg", "/Users/robertwillmot/darknet/yolov4.weights");
    // Load image0
    cv::Mat image = cv::imread("/Users/robertwillmot/Documents/University/Computer Science/3rd year/COMP6013 Dis/img_vid/pic3.jpg");

    // setup for opencv to read from a video file
    cv::VideoCapture video("/Users/robertwillmot/Documents/University/Computer Science/3rd year/COMP6013 Dis/img_vid/clip1.mp4");
    cv::Mat frame;
    cv::Rect2d bbox; //holds the bounding box of the player

    // read the first frame and check if file is a video
    if (!video.read(frame)){
        std::cerr <<"failed to read video" <<std::endl;
        return -1;
    }

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Failed to load the image!" << std::endl;
        return -1;
    }

    // Convert to HSV color space for color segmentation
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // Define the range for green color (adjust these values)
    int lowH = 30; // Example values for green
    int highH = 90;
    int lowS = 50;
    int highS = 255;
    int lowV = 50;
    int highV = 255;

    cv::Scalar lowerGreen(lowH, lowS, lowV);
    cv::Scalar upperGreen(highH, highS, highV);

    // Create a mask for green color
    cv::Mat greenMask;
    cv::inRange(hsvImage, lowerGreen, upperGreen, greenMask);

    // Apply the green mask to the original image
    cv::Mat maskedImage;
    cv::bitwise_and(image, image, maskedImage, greenMask);

    // Continue with preprocessing on the masked image
    cv::Mat grayscaleImage;
    cv::cvtColor(maskedImage, grayscaleImage, cv::COLOR_BGR2GRAY);
    cv::Mat blurredImage;
    cv::GaussianBlur(grayscaleImage, blurredImage, cv::Size(5, 5), 0);

    // Edge Detection: Canny
    cv::Mat edges;
    double lowThreshold = 50; // Adjust these values based on your image
    double highThreshold = 75;
    cv::Canny(blurredImage, edges, lowThreshold, highThreshold);

    // Detect Lines: Hough Transform
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, 80, 10); // Parameters: resolution, threshold, minLineLength, maxLineGap

    // Filter lines based on proximity to green
    int sampleDistance = 5; // Distance from the line to sample the pixels
    std::vector<cv::Vec4i> greenBoundaryLines;

    for (const auto& l : lines) {
        // Calculate a simple perpendicular vector to the line
        cv::Point2f dir(static_cast<float>(l[0] - l[2]), static_cast<float>(l[1] - l[3]));
        float norm = std::sqrt(dir.x * dir.x + dir.y * dir.y);
        dir.x /= norm;
        dir.y /= norm;

        // Sample points on both sides of the midpoint of the line
        cv::Point midPoint((l[0] + l[2]) / 2, (l[1] + l[3]) / 2);
        cv::Point samplePt1 = midPoint + cv::Point(static_cast<int>(dir.x * sampleDistance), static_cast<int>(dir.y * sampleDistance));
        cv::Point samplePt2 = midPoint - cv::Point(static_cast<int>(dir.x * sampleDistance), static_cast<int>(dir.y * sampleDistance));

        // Check if sampled points are within image bounds
        if (samplePt1.inside(cv::Rect(0, 0, image.cols, image.rows)) && samplePt2.inside(cv::Rect(0, 0, image.cols, image.rows))) {
            // Get the color of the sampled points from the green mask
            bool pt1IsGreen = (greenMask.at<uchar>(samplePt1) == 255);
            bool pt2IsGreen = (greenMask.at<uchar>(samplePt2) == 255);

            // If both sampled points are on green, it's likely a boundary line
            if (pt1IsGreen && pt2IsGreen) {
                greenBoundaryLines.push_back(l);
            }
        }
    }

    // Create a copy of the original image to draw the lines on
    cv::Mat displayImage = image.clone();

    // Draw the filtered lines on the copy of the original image
    for (const auto& l : greenBoundaryLines) {
        cv::line(displayImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    }

    // Display the image with the overlayed lines
    cv::imshow("Line Image", displayImage);

    // convert image to a blob
    cv::Mat blob;
    cv::dnn::blobFromImage(displayImage, blob, 1/255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);

    // set blob as input to network
    net.setInput(blob);

    //run forward pass to get the detections
    std::vector<cv::Mat> detections;
    net.forward(detections, net.getUnconnectedOutLayersNames());

    //detect
    float confidenceThreshold = 0.5;
    int classId;
    float confidence;
    float* data;
    for (auto& output : detections){
        for (int i = 0; i <output.rows; i++){
            data = output.ptr<float>(i);
            confidence = data[4];

            if (confidence > confidenceThreshold){
                // get class with highest score
                classId = -1;
                float maxClassScore = -1;
                for (int j=5; j < output.cols; j++){
                    float classScore = data[j];
                    if (classScore > maxClassScore){
                        maxClassScore = classScore;
                        classId = j - 5;
                    }
                }
                if (classId == 0 && maxClassScore > confidenceThreshold){ //the 0 is the class ID of a person
                    // Scale the coordinates back to the image size
                    int centerX = static_cast<int>(data[0] * displayImage.cols);
                    int centerY = static_cast<int>(data[1] * displayImage.rows);
                    int width = static_cast<int>(data[2] * displayImage.cols);
                    int height = static_cast<int>(data[3] * displayImage.rows);

                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    // Draw the bounding box
                    cv::rectangle(displayImage, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 0, 255), 3);
                }
            }
        }
    }
si
    cv::imshow("Detection Image", displayImage);

    // Exit key
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
