"""
Object Detection

Learn how to detect objects using Haar cascades and HOG detectors in OpenCV​
Object Detection
Learn how to detect objects in images and video using classical computer vision techniques including 
Haar cascades and Histogram of Oriented Gradients (HOG) detectors.

Haar Cascade Classifiers
Haar cascades are machine learning-based classifiers trained to detect specific objects. 
OpenCV comes with pre-trained models for faces, eyes, pedestrians, and more.
"""
import cv2 as cv 

CAMERA_INDEX = 0

def load_models():
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

    if face_cascade.empty():
        print("error loading cascades")
        return
    return face_cascade, eye_cascade 


def detect(frame, cascade):
    rects = cascade.detectMultiScale(frame, scaleFactor=1.3, 
                                    minNeighbors=4, minSize=(30, 30),
                                    flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]  # Convert to (x1, y1, x2, y2)
    return rects


def draw_rects(img, rects, color):
    """Draw rectangles on image"""
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)


def camera_loop(camera_index: int ):
    cap = cv.VideoCapture(camera_index)
    face_cascade, eye_cascade = load_models()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = detect(frame, eye_cascade)
        draw_rects(frame, faces, (0, 0, 255))

        cv.imshow("Face detection", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def main():
    camera_loop(CAMERA_INDEX)

if __name__ == "__main__":
    main()
