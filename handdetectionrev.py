from inference_sdk import InferenceHTTPClient
import supervision as sv
import cv2
import base64
import numpy as np

def process_frame(frame):
    # Convert the frame to a base64-encoded string
    _, buffer = cv2.imencode('.jpg', frame)
    raw_image = base64.b64encode(buffer).decode('utf-8')
    
    # Perform inference using the Roboflow API
    results = CLIENT.infer(raw_image, model_id="american-sign-language-alphabet-miocm/1")
    return results

# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="JxpBDHTQ8l44CDjW190L"
)

# Initialize webcam
cap = cv2.VideoCapture(0)  # Change 0 to the webcam index if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    results = process_frame(frame)
    
    # Extract detections using supervision
    detections = sv.Detections.from_inference(results)
    
    # Create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Annotate the frame
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    
    # Display the resulting frame
    cv2.imshow('Annotated Frame', annotated_frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()