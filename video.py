''' used to download images '''

# from simple_image_download import simple_image_download as simp

# response = simp.simple_image_download

# keywords = ["shoes"]

# for k in keywords:
#     response().download(k,200)


""" used to train """

# from ultralytics import YOLO

# model = YOLO('yolov8s-seg.pt')

# model.train(data = "shoes_data.yaml" , epochs = 50)

import cv2 as cv
from ultralytics import YOLO

model = YOLO('yolov8x.pt')
id_model = YOLO('id_card_dataset.pt')
belt_model = YOLO('belt_dataset.pt')
shoe_model = YOLO('shoes_dataset.pt')



video = cv.VideoCapture(0)
i = 5

while True:
    isTrue, frame = video.read()

    # cv.imshow('Video1',frame)
    if i == 5:
        i = 0
        persons = model.predict(frame,classes = 0 , conf = 0.5)

        for p in persons:
            boxes = p.boxes

            for box in boxes:
                b = box.xyxy[0]

                cropped_person = frame[int(b[1]) : int(b[3]) , int(b[0]) : int(b[2])]

                ids = id_model.predict(cropped_person )
                # cv.waitKey(0)
                belts = belt_model.predict(cropped_person  )
                # cv.waitKey(0)
                shoes = shoe_model.predict(cropped_person)
                # cv.waitKey(0)

                if(len(ids[0].boxes.cls) >= 1 and len(belts[0].boxes.cls) >= 1 and len(shoes[0].boxes.cls) >= 1):
                    cv.rectangle(frame,(int(b[0]) , int(b[1])) , (int(b[2]) , int(b[3])) , (0,255,0) , thickness = 3)
                else:
                    cv.rectangle(frame , (int(b[0]) , int(b[1])) , (int(b[2]) , int(b[3])) , (0,0,225) , thickness = 3)

            cv.imshow("Video" , frame)
            cv.waitKey(1)

    else:
        i += 1

