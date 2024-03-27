#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Read the image file
    Mat image = imread("image.jpg");

    // Check if the image was successfully loaded
    if (image.empty()) {
        std::cerr << "Error: Unable to load image" << std::endl;
        return 1;
    }

    // Display the image
    imshow("Image", image);

    // Wait for a key press
    waitKey(0);

    // Close all OpenCV windows
    destroyAllWindows();

    return 0;
}
