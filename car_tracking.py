import cv2

# Our image
img_file = 'car_image.jpg'

# Pre-trained car classifier
classifier_file = 'car_detector.xml'

# Create opencv image
img = cv2.imread(img_file)

# Create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)


# Display the image with car
cv2.imshow('Car Detector', img)
cv2.waitKey()





print("Code is ok")