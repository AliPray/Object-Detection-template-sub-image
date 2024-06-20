import cv2
import numpy as np
from operator import itemgetter


#func to read the oject's picture as a template from the cropped image
def object_capture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    canny_output = cv2.Canny(gaussian, 20, 150)
    dilate = cv2.dilate(canny_output, None, iterations=1)
    erode = cv2.erode(dilate, None, iterations=1)
    contours, hierarchy = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corner_list = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 700:
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            area_detect_gray = gray[y-5: y+h+5, x-5: x+w+5].copy()
            corners = cv2.goodFeaturesToTrack(area_detect_gray, 100, 0.05, 50)
            corners = np.int0(corners)
            cv2.rectangle(image, (x-2, y-2), (x+w+2, y+h+2), (0, 0, 255), 2)
            for corner in corners:
                cx, cy = corner.ravel()
                cv2.circle(image, (cx+x, cy+y), 4, (255, 0, 0), -1)
                corner_list.append([cx+x, cy+y])
            corner_list.sort(key=itemgetter(1))
    window_name = 'Object'
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 40, 30)
    cv2.imshow(window_name, image)
    return corner_list

#func to detect the template object inside the sub images ( test images)
def subimage(img, num, length, main_object_distance):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applying the filter
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

    # Object Detection
    canny_output = cv2.Canny(gaussian, 50, 150)

    dilate = cv2.dilate(canny_output, None, iterations=1)
    erode = cv2.erode(dilate, None, iterations=1)

    # Find all contours
    contours, hierarchy = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    corner_list = []
    # loop through all the object contour
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # only select object with area larger than 700
        if area > 700:
            # Use OpenCV boundingRect function to get the details of the contour
            x, y, w, h = cv2.boundingRect(cnt)
            area_detect_gray = gray[y - 5: y + h + 5, x - 5: x + w + 5].copy()

            corners = cv2.goodFeaturesToTrack(area_detect_gray, 100, 0.05, 50)
            if corners is not None:
                corners = np.int0(corners)
            else:
                print("No corners detected")
                continue

            if len(corners) != length:
                continue

            cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 2)

            temp = []
            for corner in corners:
                cx, cy = corner.ravel()
                cv2.circle(img, (cx + x - 5, cy + y - 5), 4, (255, 0, 0), -1)
                temp.append([cx + x - 5, cy + y - 5])

            corner_list.append(temp)

    corner_list.sort(key=itemgetter(0))

    for i, corners in enumerate(corner_list):
        if len(corners) > 2:
            object_distance = calculate_distance(corners)
            fit = evaluate_distance(main_object_distance, object_distance)
            text = f"{fit}% Similarity"
            cv2.putText(img, text, corners[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (124,252,0), 2, cv2.LINE_AA)
            print(f"the Euclidean distance in test image_{num}: {object_distance}")
        else:
            object_distance = calculate_distance(corners[0])
            fit = evaluate_distance(main_object_distance, object_distance)
            text = f"{fit}% Similarity"
            cv2.putText(img, text, corners[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            print(f"the Euclidean distance in test image_{num}: {object_distance}")

    name = f"Image{num}"
    cv2.imshow(name, img)

    return corner_list


#func to calculate the Euclidean distance ( standard deviation of the corner points detectedof the object)
def euclidean_distance(x, y):
    return np.sqrt(np.square(x[0] - x[1]) + np.square(y[0] - y[1]))

def calculate_average(lst):
    return sum(lst) / len(lst)

#func to calculate distance from the corner points detected
def calculate_distance(corner_points):
    result_list = []
    for num in range(len(corner_points)):
        if(num == len(corner_points) - 1):
            result = euclidean_distance(corner_points[num], corner_points[0])
            result_list.append(result)
        else:
            result = euclidean_distance(corner_points[num], corner_points[num+1])
            result_list.append(result)
    avg = calculate_average(result_list)
    return avg

#func to evaluate the distance of the object's in the test images against the template's distance
def evaluate_distance(template, subimage_avg):
    evaluation = (subimage_avg / template) * 100
    return round(evaluation, 2)

object_distance = ''

#reading images
object = cv2.imread('template.jpeg')
image1 = cv2.imread('test1.jpg')
image2 = cv2.imread('test2.jpg')
image3 = cv2.imread('test3.jpg')
image4 = cv2.imread('test4.jpg')

#running the functions mentioned above
corner_points = object_capture(object)
corner_count = len(corner_points)
object_Distance = calculate_distance(corner_points)

print('the Euclidiean Distance of template: ',object_Distance)

corner_points_1 = subimage(image1,1,corner_count,object_Distance)
corner_points_2 = subimage(image2,2,corner_count,object_Distance)
corner_points_3 = subimage(image3,3,corner_count,object_Distance)
corner_points_4 = subimage(image4,4,corner_count,object_Distance)

cv2.waitKey(0)
cv2.destroyAllWindows