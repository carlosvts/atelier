from pathlib import Path
import time

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


CAMERA_INDEX = 0

FACE_MODEL_PATH = Path("models/face_landmarker.task")
HAND_MODEL_PATH = Path("models/hand_landmarker.task")


def create_face_landmarker(model_path: Path):
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.FaceLandmarker.create_from_options(options)


def create_hand_landmarker(model_path: Path):
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.HandLandmarker.create_from_options(options)


def frame_to_mp_image(frame_bgr):
    frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def draw_normalized_landmarks(frame, landmarks_list, color=(0, 255, 0), radius=1):
    if not landmarks_list:
        return

    height, width, _ = frame.shape

    for landmarks in landmarks_list:
        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            if 0 <= x < width and 0 <= y < height:
                cv.circle(frame, (x, y), radius, color, -1)


def draw_blendshapes(frame, face_result):
    if not face_result.face_blendshapes:
        return

    blendshapes = face_result.face_blendshapes[0]
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


def draw_hands_info(frame, hand_result):
    if not hand_result.handedness:
        return

    y = frame.shape[0] - 60

    for i, handedness in enumerate(hand_result.handedness):
        if not handedness:
            continue

        category = handedness[0]
        text = f"hand {i}: {category.category_name} {category.score:.2f}"

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


def draw_debug_info(frame, face_result, hand_result, fps: float):
    face_count = len(face_result.face_landmarks) if face_result.face_landmarks else 0
    hand_count = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0

    text = f"FPS: {fps:.1f} | faces: {face_count} | hands: {hand_count}"

    cv.putText(
        frame,
        text,
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )


def camera_loop(camera_index: int):
    if not FACE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {FACE_MODEL_PATH}")

    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {HAND_MODEL_PATH}")

    cap = cv.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    start_time = time.monotonic()
    previous_time = time.monotonic()
    fps = 0.0

    try:
        with (
            create_face_landmarker(FACE_MODEL_PATH) as face_landmarker,
            create_hand_landmarker(HAND_MODEL_PATH) as hand_landmarker,
        ):
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

                face_result = face_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                hand_result = hand_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                draw_normalized_landmarks(
                    frame,
                    face_result.face_landmarks,
                    color=(0, 255, 0),
                    radius=1,
                )

                draw_normalized_landmarks(
                    frame,
                    hand_result.hand_landmarks,
                    color=(255, 0, 0),
                    radius=2,
                )

                draw_blendshapes(frame, face_result)
                draw_hands_info(frame, hand_result)
                draw_debug_info(frame, face_result, hand_result, fps)

                cv.imshow("mediapipe debugger", frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv.destroyAllWindows()


def main():
    camera_loop(CAMERA_INDEX)


if __name__ == "__main__":
    main()