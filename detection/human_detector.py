import cv2
from ultralytics import YOLO

video_path = "data/people_walking.mp4" #video path

model = YOLO("yolov8n.pt") #"n" so it is nano model, faster

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video could not open!")
    exit()

while True:
    ret, frame =cap.read() #ret is a boolean that indicates if video opened or not

    if not ret:
        break #they are no frame anymore

    # Since model(frame) return a list of
    # results (even though in our program it will return only one)
    # we need to return first element of the list
    results = model(frame)[0] 

    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()