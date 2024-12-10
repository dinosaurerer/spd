from ultralytics import YOLO
import cv2

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Define path to the image file
source = r"D:\PyCharmWorkSpace\spd\spd\YOLO\bus.jpg"

# Run inference on the source
results = model(source)  # list of Results objects
cv2.imshow('bus', results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()