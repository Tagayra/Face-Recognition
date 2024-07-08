import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")

# Open the video file
video_path = 0
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model(frame, conf=0.5)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                conf = round(box.conf[0].item(), 2)
                name = r.names[box.cls[0].item()]
                x1, y1, x2, y2 = int(b[0].item()), int(b[1].item()), int(b[2].item()), int(b[3].item())

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Tagayra {str(conf)}', (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 105, 180, 100), 1)



        # Display the annotated frame
        cv2.imshow(" Trackers", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
