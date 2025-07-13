import cv2
from ultralytics import YOLO

video_path = "data/people_walking.mp4" #video path

 #"n" so it is nano model, faster. 
 #It is trained by COCO dataset that includes 80 different labels 
 #I used nano model but you can use small, medium,...
model = YOLO("yolov8n.pt")

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
        if model.names[cls_id] == "person": # model.names is a dict for ex. {id1: "person", id2: "biycycle", id3: "car"..}

            conf=float(box.conf[0]) #this is the probability of that object
            if conf > 0.55: #this is a threshold value if prob > 0.50 show it!
                x1, y1, x2, y2 = map(int, box.xyxy[0]) #coordinates of the box take it as integer
                label = f"{model.names[cls_id]}, {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 117, 44), 2)
                cv2.putText(frame, label, (x1, (y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 117, 44), 2)


    (text_width, text_height), _ = cv2.getTextSize(f"Human Count: {len(results.boxes)}", cv2.FONT_HERSHEY_COMPLEX, 1, 2)
    x = frame.shape[1] - text_width - 10
    y = text_height + 10
    cv2.putText(frame, f"Human Count: {len(results.boxes)}",(x, y) , cv2.FONT_HERSHEY_COMPLEX, 1, (255, 117, 44) , 2)

    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()