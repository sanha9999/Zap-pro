import torch
import cv2 
import time
import numpy as np
import argparse
import os

from kalmanfilter import Sort

os.environ['KMP_DUPLICATE_LIB_OK']='True' # OMP: Error #15

def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--video_path')

    config = p.parse_args()

    return config

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', verbose=False)
    model.eval()
    return model

def detection(model, min_confidence, frame):
    start_time = time.time()
    font = cv2.FONT_HERSHEY_DUPLEX
    
    img = cv2.resize(frame, None, fx=0.8, fy=0.8)
    
    outs = model(img)
    outs = outs.xyxy[0]
    

    for detection in outs:
        
        confidence = detection[4]
        class_id = int(detection[5])

        if class_id == 0 and confidence > min_confidence:
            x_min = int(detection[0])
            y_min = int(detection[1])
            x_max = int(detection[2])
            y_max = int(detection[3])
               
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
            cv2.putText(img, "person", (x_min, y_min - 5), font, 1, (255, 255, 255), 1)

    #process_time = end_time - start_time
    #print("=== {:.3f} seconds".format(process_time))
    cv2.imshow("test", img)

def main(config):
    video_path = config.video_path
    min_confidence = 0.4
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = load_model()
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened:
        print('--@@@@@ ERROR @@@@@--')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--@@@@@ No captured frame @@@@@--')
            break
        else:
            detection(model , min_confidence, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    config = define_argparser()
    main(config)

