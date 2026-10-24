
import time
import cv2
from ultralytics import YOLO


MODEL_NAME = "yolo26n.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.35
IMAGE_SIZE = 640


def load_model(model_name):
    """Load the pretrained YOLO detection model."""
    print(f"Loading model: {model_name}")
    return YOLO(model_name)


def open_camera(camera_index):
    """Open the webcam and check that it is available."""
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions "
            "or try CAMERA_INDEX = 1."
        )

    return camera


def detect_objects(model, frame):
    """Run object detection on one webcam frame."""
    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False
    )
    return results[0]


def main():
    model = load_model(MODEL_NAME)
    camera = open_camera(CAMERA_INDEX)

    previous_time = time.perf_counter()

    print("Webcam started. Press Q or ESC to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a frame from webcam.")
                break

            result = detect_objects(model, frame)

            # Draw detection boxes and labels.
            output_frame = result.plot()

            # Count detected objects in the current frame.
            object_count = len(result.boxes)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous_time = current_time

            cv2.putText(
                output_frame,
                f"Objects: {object_count}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                output_frame,
                f"FPS: {fps:.1f}",
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("YOLO Real-Time Object Detection", output_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Application error: {error}")
