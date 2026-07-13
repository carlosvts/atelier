from pathlib import Path
import time

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


CAMERA_INDEX = 0
MODEL_PATH = Path("models/face_landmarker.task")


def create_face_landmarker(model_path: Path):
    base_options = python.BaseOptions(model_asset_path=str(model_path))

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.FaceLandmarker.create_from_options(options)


def frame_to_mp_image(frame_bgr):
    frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def draw_face_landmarks(frame, result):
    if not result.face_landmarks:
        return

    height, width, _ = frame.shape

    for face_landmarks in result.face_landmarks:
        for landmark in face_landmarks:
            # since landmarks info is normalized, we need to multiply by width and height to get the real coordinates
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            if 0 <= x < width and 0 <= y < height:
                cv.circle(frame, (x, y), 1, (0, 255, 0), -1)


def draw_blendshapes(frame, result):
    if not result.face_blendshapes:
        return

    blendshapes = result.face_blendshapes[0]
    top_blendshapes = sorted(
        blendshapes,
        key=lambda category: category.score,
        reverse=True,
    )[:6]

    y = 60

    for category in top_blendshapes:
        text = f"{category.category_name}: {category.score:.2f}"
        cv.putText(
            frame,
            text,
            (20, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        y += 24


def draw_debug_info(frame, result, fps: float):
    face_count = len(result.face_landmarks) if result.face_landmarks else 0

    cv.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )

    cv.putText(
        frame,
        f"faces: {face_count}",
        (20, frame.shape[0] - 20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )


def camera_loop(camera_index: int):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Download face_landmarker.task into the models/ directory."
        )

    cap = cv.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    start_time = time.monotonic()
    previous_time = time.monotonic()
    fps = 0.0

    try:
        with create_face_landmarker(MODEL_PATH) as landmarker:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                now = time.monotonic()
                delta = now - previous_time
                previous_time = now

                if delta > 0:
                    fps = 1.0 / delta

                timestamp_ms = int((now - start_time) * 1000)

                mp_image = frame_to_mp_image(frame)

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                draw_face_landmarks(frame, result)
                draw_blendshapes(frame, result)
                draw_debug_info(frame, result, fps)

                cv.imshow("mediapipe face debugger", frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv.destroyAllWindows()


def main():
    camera_loop(CAMERA_INDEX)


if __name__ == "__main__":
    main()