import cv2
import argparse
import supervision as sv
import numpy as np
from ultralytics import YOLO

ZONE_POLYGON = np.array([
    [0, 0],             # Upper left
    [0.5, 0],          # Upper right
    [0.5, 1],        # Bottom right
    [0, 1]            # Bottom left
])

# Resolution function
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",      # name of value
        default=[1280, 720],        # added value
        nargs=2,                    # number of value
        type=int                    # type of value
    )
    args = parser.parse_args()
    return args

# Main callable function
def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Capture the frame
    cap = cv2.VideoCapture(0)       # 0 means #1 device used for object detection 
    
    # Setting the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Insert the model used
    model = YOLO("yolov8n.pt")

    # Add Bounding Box
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # Object counting in certain frame
    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
        )

    # Accessing the webcam
    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id == 39]           # Only detect certain class in whole frame
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _ in detections      # there are 6 keys so _ represent keys that aren't used
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("Object Detection", frame)

        # Show the frame size (Height, Width, Color)
        #print(frame.shape)
        #break

        # Exiting the while loop
        if (cv2.waitKey(30) == 27):     # 30 is delay in milisecond, 27 is esc button in ASCII        
            break

if __name__ == "__main__":
    main()