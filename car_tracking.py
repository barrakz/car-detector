import cv2

# Our video
img_file = 'car_image2.jpg'
video = cv2.VideoCapture('car_camera.mp4')

# Pre-trained car classifier
car_tracker_file  = 'car_detector.xml'
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)

# Run
while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Car Detector', frame)
    cv2.waitKey(1)
