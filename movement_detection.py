import cv2

def main():
    cap = cv2.VideoCapture(0)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 1200:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.drawContours(frame1, contours, -1, (0, 255, 0), 1)
            cv2.putText(frame1, "Number of movement detect : {}".format(sum(map(lambda i: cv2.contourArea(i) > 1200, contours))), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1)

        cv2.imshow("feed", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(40) == 27:
            break
    cap.release()

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()