import cv2

url = 'rtsp://admin:1234@10.120.0.27:8554/live'
# rtsp://admin:1234@10.90.128.196:8081/live
cap = cv2.VideoCapture(url)

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
