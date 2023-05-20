import cv2

# Our image
img_file = 'car_image2.jpg'

# Our video
video = cv2.VideoCapture('')

# Pre-trained car classifier
classifier_file = 'car_detector.xml'

# Create opencv image
img = cv2.imread(img_file)


# Convert to greyscale
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect cars
cars = car_tracker.detectMultiScale(black_n_white) 

# Draw rectangles around the cars

for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the image with car
cv2.imshow('Car Detector', img)
cv2.waitKey()


print("Code is ok")
 