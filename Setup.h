#ifndef STATEYE2_SETUP_H
#define STATEYE2_SETUP_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Setup {
public:
    Setup(const std::string& videoPath);
    cv::Scalar getSelectedColourHSV() const;
    std::vector<cv::Point> getSelectedPoints() const;
    void handleDistanceInput();
    void selectColourAndPoints();

    float getBaseScaleFactorWidth() const;

    float getBaseScaleFactorLengthNear() const;

    float getBaseScaleFactorLengthFar() const;

    float getBaseScaleFactorGoal() const;

    float getBaseScaleFactorPenaltyBox() const;


private:
    cv::VideoCapture video;
    cv::Mat frame;
    std::vector<cv::Point> selectedPoints;
    cv::Scalar selectedColourHSV;

    enum SelectionState {
        COLOUR_SELECTION,
        POINT_SELECTION,
        DISTANCE_INPUT
    };


    size_t currentInstructionIndex = 0;
    std::vector<std::string> instructions;

    static void onMouse(int event, int x, int y, int flags, void* userdata);
    void displayInstructions();

    void calculateScaleFactors(float distanceHalfwayToCorner, float distanceHalfwayAcross,
                               float distancePenaltyBox);

    float baseScaleFactorWidth = 0.0f;
    float baseScaleFactorLengthNear = 0.0f;
    float baseScaleFactorLengthFar = 0.0f;
    float baseScaleFactorGoal = 0.0f;
    float baseScaleFactorPenaltyBox = 0.0f;
    SelectionState currentState;

    void handleColourSelection(int y, int x);

    void advanceInstruction();
};

#endif //STATEYE2_SETUP_H
