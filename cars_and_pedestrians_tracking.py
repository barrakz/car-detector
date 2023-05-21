import cv2

# Our video
img_file = 'car_image2.jpg'
video = cv2.VideoCapture('car_camera.mp4')

# Car and pedestrian classifier files
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

# Create car and pedestrian classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# Run
while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Car Detector', frame)
    cv2.waitKey(1)
