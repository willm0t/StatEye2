#include "Setup.h"
#include <iostream>

Setup::Setup(const std::string& videoPath) : video(videoPath), currentState(SelectionState::COLOUR_SELECTION) {
    instructions = {
            "Select the position of the near side corner flag.",
            "Select the position of the near side halfway line.",
            "Select the position of the far side halfway line.",
            "Select the position of the far side corner flag.",
            "Select the closest goalpost.",
            "Select the furthest goalpost.",
            "Select the edge of the penalty box closest to you.",
            "Select the edge of the penalty box furthest away from you."
    };
    if (!video.isOpened()) {
        throw std::runtime_error("Failed to open video for setup.");
    }
}

void Setup::selectColourAndPoints() {
    if (!video.read(frame)) {
        throw std::runtime_error("Failed to read first frame from video.");
    }

    cv::namedWindow("Setup - Select Colour and Points", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Setup - Select Colour and Points", onMouse, this);

    std::cout << "Please select the team's kit colour by clicking on a player." << std::endl;
    cv::imshow("Setup - Select Colour and Points", frame);
    cv::waitKey(0); // Wait for the user to select points and colour

    cv::destroyWindow("Setup - Select Colour and Points");
}

void Setup::onMouse(int event, int x, int y, int flags, void* userdata) {
    Setup* self = reinterpret_cast<Setup*>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (self->currentState == SelectionState::COLOUR_SELECTION) {
            // Color selection logic...
            self->currentState = SelectionState::POINT_SELECTION;
            self->advanceInstruction(); // Ensure this advances to first point selection instruction
        } else if (self->currentState == SelectionState::POINT_SELECTION) {
            self->selectedPoints.push_back(cv::Point(x, y));
            if (self->currentInstructionIndex < self->instructions.size()) {
                // If more points to select, advance to next instruction
                self->advanceInstruction();
            } else {
                // If all points are selected, prepare for distance input
                std::cout << "All selections completed. Please proceed to distance input." << std::endl;
                self->currentState = SelectionState::DISTANCE_INPUT;
            }
        }
    }
}

void Setup::handleColourSelection(int y, int x) {
    cv::Vec3b colour = frame.at<cv::Vec3b>(y, x);
    cv::Mat colourMat(1, 1, CV_8UC3, colour);
    cv::cvtColor(colourMat, colourMat, cv::COLOR_BGR2HSV);
    selectedColourHSV = colourMat.at<cv::Vec3b>(0, 0);

    std::cout << "Kit colour selected." << std::endl;
    currentState = SelectionState::POINT_SELECTION; // Proceed to point selection
}

void Setup::advanceInstruction() {
    if (currentState == SelectionState::POINT_SELECTION && currentInstructionIndex < instructions.size()) {
        std::cout << instructions[currentInstructionIndex++] << std::endl;
    }
}


void Setup::handleDistanceInput() {
    float distanceHalfwayToCorner, distanceHalfwayAcross, distancePenaltyBox;

    std::cout << "Enter the distance from near side halfway line to near side corner flag (in meters): ";
    std::cin >> distanceHalfwayToCorner;

    std::cout << "Enter the distance across the field from near side halfway line to far side halfway line (in meters): ";
    std::cin >> distanceHalfwayAcross;

    std::cout << "Enter the distance from the edge of the penalty box closest to you to the furthest (in meters): ";
    std::cin >> distancePenaltyBox;

    // Now that we have the distances, we calculate the scale factors
    calculateScaleFactors(distanceHalfwayToCorner, distanceHalfwayAcross, distancePenaltyBox);
}

void Setup::displayInstructions() {
    if (currentInstructionIndex < instructions.size()) {
        std::cout << instructions[currentInstructionIndex++] << std::endl;
    }
}

void Setup::calculateScaleFactors(float distanceHalfwayToCorner, float distanceHalfwayAcross,
                                  float distancePenaltyBox) {
    float pixelDistanceHalfwayToCornerNear = cv::norm(selectedPoints[1] - selectedPoints[2]);
    float pixelDistanceHalfwayAcross = cv::norm(selectedPoints[2] - selectedPoints[3]);
    float pixelDistanceHalfwayToCornerFar = cv::norm(selectedPoints[3] - selectedPoints[4]);
    float pixelDistanceGoal = cv::norm(selectedPoints[5] - selectedPoints[6]);
    float pixelDistancePenaltyBox = cv::norm(selectedPoints[7] - selectedPoints[8]);

    // Calculate scale factors based on the real-world measurements and pixel distances
    baseScaleFactorWidth = distanceHalfwayAcross / pixelDistanceHalfwayAcross;
    baseScaleFactorLengthNear = distanceHalfwayToCorner / pixelDistanceHalfwayToCornerNear;
    baseScaleFactorLengthFar = distanceHalfwayToCorner / pixelDistanceHalfwayToCornerFar;
    baseScaleFactorGoal = 7.32f / pixelDistanceGoal; // Assuming 7.32m as the standard goal width
    baseScaleFactorPenaltyBox = distancePenaltyBox / pixelDistancePenaltyBox;

    // Output for debugging
    std::cout << "Scale factors calculated.\n";
    std::cout << "Width Scale Factor: " << baseScaleFactorWidth << "\n";
    std::cout << "Length Near Scale Factor: " << baseScaleFactorLengthNear << "\n";
    std::cout << "Length Far Scale Factor: " << baseScaleFactorLengthFar << "\n";
    std::cout << "Goal Scale Factor: " << baseScaleFactorGoal << "\n";
    std::cout << "Penalty Box Scale Factor: " << baseScaleFactorPenaltyBox << std::endl;
}

float Setup::getBaseScaleFactorPenaltyBox() const {
    return baseScaleFactorPenaltyBox;
}

float Setup::getBaseScaleFactorGoal() const {
    return baseScaleFactorGoal;
}

float Setup::getBaseScaleFactorLengthFar() const {
    return baseScaleFactorLengthFar;
}

float Setup::getBaseScaleFactorLengthNear() const {
    return baseScaleFactorLengthNear;
}

float Setup::getBaseScaleFactorWidth() const {
    return baseScaleFactorWidth;
}

cv::Scalar Setup::getSelectedColourHSV() const {
    return selectedColourHSV;
}

std::vector<cv::Point> Setup::getSelectedPoints() const {
    return selectedPoints;
}



