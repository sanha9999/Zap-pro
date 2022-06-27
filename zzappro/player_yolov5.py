import torch
import cv2 
import time
import numpy as np
import argparse
import os

from kalmanfilter import *

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

def detection(model, mot_tracker, min_confidence, frame):
    start_time = time.time()
    font = cv2.FONT_HERSHEY_DUPLEX
    
    img = cv2.resize(frame, None, fx=0.8, fy=0.8)
    
    outs = model(img)
    outs = outs.pred[0].cpu().numpy()
    track_bbs_ids = mot_tracker.update(outs)
    
    for i in range(len(track_bbs_ids.tolist())):
        
        coords = track_bbs_ids.tolist()[i]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        name_idx = int(coords[4])
        name = "Id : {}".format(str(name_idx))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(img, name, (x1, y1 - 5), font, 1, (255, 255, 255), 1)

    
    cv2.imshow("test", img)

def main(config):    
    video_path = config.video_path
    min_confidence = 0.5
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = load_model()
    model.to(device)

    mot_tracker = Sort()

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
            detection(model, mot_tracker, min_confidence, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    config = define_argparser()
    main(config)

