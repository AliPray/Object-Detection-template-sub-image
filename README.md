# Object Detection Using Edge Detection Techniques

## Overview

This project demonstrates an object detection method using various edge detection techniques. The code captures a template object's image and then detects and compares this object in test images. The detection is based on contour analysis and corner detection. The results are evaluated by calculating the Euclidean distance of detected corner points and comparing the template object with objects in test images for similarity.

## Implementation Details

### Object Detection Workflow

1. **Template Object Capture**: The template object is captured from a given image using edge detection and contour analysis.
2. **Test Image Processing**: Each test image is processed to detect the template object using the same edge detection and contour analysis techniques.
3. **Distance Calculation**: The Euclidean distance between detected corner points is calculated for both the template and objects in test images.
4. **Similarity Evaluation**: The similarity between the template object and objects in test images is evaluated based on the calculated distances.

### Key Functions

- `object_capture(image)`: Captures the template object from an input image and detects its corner points.
- `subimage(img, num, length, main_object_distance)`: Detects the template object in test images and evaluates the similarity.
- `euclidean_distance(x, y)`: Computes the Euclidean distance between two points.
- `calculate_average(lst)`: Calculates the average value of a list.
- `calculate_distance(corner_points)`: Calculates the average Euclidean distance of corner points.
- `evaluate_distance(template, subimage_avg)`: Evaluates the similarity between the template and detected objects in test images.

### Steps to Run the Code

1. **Read Template and Test Images**: Load the template image and test images using `cv2.imread`.
2. **Capture Template Object**: Use `object_capture` to detect and capture the template object from the template image.
3. **Process Test Images**: Apply `subimage` to each test image to detect the template object and evaluate the similarity.
4. **Display Results**: The detected objects and similarity scores are displayed using OpenCV's `imshow`.

## Dependencies

- Python 3.x
- OpenCV
- NumPy

## Results

The results include:
- Processed images with detected objects highlighted.
- Similarity scores indicating how closely the detected objects in test images match the template object.
