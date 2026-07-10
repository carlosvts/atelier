import cv2 as cv

def camera_loop(camera_index): 
    cap: cv.VideoCapture = cv.VideoCapture(camera_index)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        edges = cv.Canny(frame, 50, 100)
        edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        blurred = cv.GaussianBlur(frame, (5,5), 0)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cv.putText(blurred, "hello opencv!", (10, 30),
            cv.FONT_HERSHEY_SIMPLEX, 2, (245, 245, 245), 3)

        height, width = frame.shape[:2]
        cv.circle(blurred, (height//2, width//2), 50, (255, 0, 0), 1)
        
        #                                           BGR  
        cv.rectangle(edges, (10, 10), (100, 100), (255, 255, 255), 2)
        cv.rectangle(edges_bgr, (10, 10), (100, 100), (0, 0, 255), 2)
         
        cv.imshow("edges", edges)
        cv.imshow("blurry", blurred)
        cv.imshow("edges bgr", edges_bgr)
        cv.imshow("gray", gray)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def main():
    camera_loop(0)

if __name__ == "__main__":
    main()
