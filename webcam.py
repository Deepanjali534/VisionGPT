import cv2
import torch
from PIL import Image
from visiongpt.pipeline.detector import detect
from visiongpt.utils.visualizer import visualize
import config
import tempfile
import os


FRAME_SKIP = 5

def draw_boxes_on_frame(frame, detections):
    """Draw bounding boxes directly on the OpenCV frame."""
    COLORS = [
        (255, 0, 0),    
        (0, 255, 0),    
        (0, 0, 255),    
        (255, 0, 255),  
        (0, 255, 255),  
        (255, 165, 0), 
    ]

    for i, det in enumerate(detections):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = f"{det['label']} {det['score']:.2f}"

        # draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # draw black background behind label
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(label) * 10, y1), (0, 0, 0), -1)

        # draw label text
        cv2.putText(frame, label, (x1 + 2, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return frame


def run_webcam():
    print("Starting webcam... press Q to quit")
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Error: could not open webcam")
        return

    detections = []   # store last detections
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: could not read frame")
            break

        frame_count += 1

        # only run DETR every FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:
            # convert OpenCV frame (BGR) to PIL image (RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # save to temp file so detect() can read it
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                pil_image.save(tmp.name)
                tmp_path = tmp.name

            # run DETR detection
            try:
                detections = detect(tmp_path)
            except Exception as e:
                print(f"Detection error: {e}")

            # clean up temp file
            os.unlink(tmp_path)

        # draw latest detections on every frame
        frame = draw_boxes_on_frame(frame, detections)

        # show object count on screen
        cv2.putText(frame, f"Objects: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # show the frame
        cv2.imshow("VisionGPT — Live Detection (Q to quit)", frame)

        # press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stopped.")


if __name__ == "__main__":
    run_webcam()