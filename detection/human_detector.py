import cv2
from ultralytics import YOLO

video_path = "data/people_walking.mp4" #video path

model = YOLO("yolov8n.pt") #"n" so it is nano model, faster. It is trained by COCO dataset that includes 80 different labels

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

    for box in results.boxes:
        cls_id = int(box.cls[0]) #take the id of the object
        conf=float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{model.names[cls_id]}, {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 117, 44), 2)
        cv2.putText(frame, label, ((x1+20), (y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 117, 44), 2)



    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()