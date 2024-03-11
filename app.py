import cv2
import supervision as sv
from ultralytics import YOLO

path = r"madrid vs city.mp4"
box_annotator = sv.BoxAnnotator()
pre_trained_model = YOLO(r"best.pt")


def Test_Video(path):
    
    camera = cv2.VideoCapture(path)
    
    while camera.isOpened():
        
        success, frame = camera.read()
        
        if success:
            
            results = pre_trained_model(frame)[0]
            
            detections = sv.Detections.from_ultralytics(results)
            
            labels = [
               f"{pre_trained_model.model.names[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            
            annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
            
            cv2.imshow("Output", annotated_frame)

            if cv2.waitKey(1) == ord("q"):
                break

        else:
            raise(r"Can't access camera/video")
            
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Test_Video(path)