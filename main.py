import cv2
import numpy as np

# Threshold for detection and NMS
thres = 0.45
nms_threshold = 0.2

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load class names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the model configuration and weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Start capturing video and detecting objects
while True:
    success, img = cap.read()
    if not success:
        print("Failed to read video stream.")
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    # Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    # If there are any detections, process them
    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classIds[i] - 1].upper(), (x + 10, y + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Show the output
    cv2.imshow("Output", img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
