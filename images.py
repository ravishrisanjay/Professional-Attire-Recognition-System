import cv2 as cv
from ultralytics import YOLO

model = YOLO('yolov8x.pt')
id_model = YOLO('id_card_dataset.pt')
belt_model = YOLO('belt_dataset.pt')
shoe_model = YOLO('shoes_dataset.pt')

temp_img = cv.imread('image1.jpg')
output_img = cv.imread('image1.jpg')

id_model.predict(temp_img , show = True)
cv.waitKey(0)
belt_model.predict(temp_img,show = True)
cv.waitKey(0)
shoe_model.predict(temp_img , show = True)
cv.waitKey(0) 

persons = model.predict(temp_img, classes = 0 , show = True , conf = 0.5 , save = True)

for p in persons :
    boxes = p.boxes

    for box in boxes:
        b = box.xyxy[0]

        cropped_person = temp_img[int(b[1]) : int(b[3]) , int(b[0]) : int(b[2])]

        ids = id_model.predict(cropped_person , show = True)
        cv.waitKey(0)
        belts = belt_model.predict(cropped_person  , show = True)
        cv.waitKey(0)
        shoes = shoe_model.predict(cropped_person , show = True)
        cv.waitKey(0)

        if(len(ids[0].boxes.cls) >= 1 and len(belts[0].boxes.cls) >= 1 and len(shoes[0].boxes.cls) >= 1):
            cv.rectangle(output_img, (int(b[0]) , int(b[1])) , (int(b[2]) , int(b[3])) , (0,255,0) , thickness = 3)
        else:
            cv.rectangle(output_img, (int(b[0]) , int(b[1])) , (int(b[2]) , int(b[3])) , (0,0,255) , thickness = 3)


cv.imshow("Final Output",output_img)
cv.waitKey(0)