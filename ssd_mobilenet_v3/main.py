import cv2 # pip install opencv-python

config_file = 'ssd_mobilenet_v3\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'ssd_mobilenet_v3\\frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# output folder and the id start number change these!!!!!!
outputFolder = "reid-data\\newdataset\\NewData\\query"
id = 0

classLabels = []
file_name = "ssd_mobilenet_v3\\labels.txt"
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip("\n").split("\n")
print(classLabels)

model.setInputSize(320, 320) # image resolution
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5)) # mobilenet => [-1,1]
model.setInputSwapRB(True)

cap = cv2.VideoCapture("ssd_mobilenet_v3\\test_2.mp4")
font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN

if cap.isOpened():
    while True:
        ret, frame = cap.read()

        ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.6)

        if len(ClassIndex) != 0:
            for ClassInd, conf, boxes in zip(
                ClassIndex.flatten(), confidence.flatten(), bbox
            ):

                # crop the photo and save it in the output folder
                x, y, x1, y1 = boxes

                # frame[height:height+height, width:width+width] with additonal padding
                padding = 5
                crop_img = frame[y - padding:y1 + y + padding,
                                 x - padding:x1 + x + padding]
                id += 1
                image_name = outputFolder + str(id) + ".jpg"

                # Extract image
                cv2.imwrite(image_name, crop_img)
                print(image_name)

                # Draw box
                if ClassInd == 1:
                    cv2.rectangle(frame, boxes, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Person Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
cv2.destroyAllWindows()
