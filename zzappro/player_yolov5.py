from turtle import dot
import torch
import cv2 
import time
import numpy as np
import argparse
import os

from kalmanfilter import *
from mouse_event import *

os.environ['KMP_DUPLICATE_LIB_OK']='True' # OMP: Error #15

def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--video_path')

    config = p.parse_args()

    return config

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', verbose=False)
    model.eval()
    return model

def detection(model, mot_tracker, min_confidence, frame, field_img, point_list, frame_size):
    start_time = time.time()
    font = cv2.FONT_HERSHEY_DUPLEX
    
    img = cv2.resize(frame, None, fx=0.5, fy=0.4)
    field_img = cv2.resize(field_img, (frame_size[0], frame_size[1]))

    point_gt = point_list[0]
    point_field = point_list[1]

    point_gt = point_gt.astype(np.float32)
    point_field = point_field.astype(np.float32)

    matrix = cv2.getPerspectiveTransform(point_gt, point_field)
    
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

        x = x1 + int((x2 - x1) / 2)
        y = y2 - int((y1 - y2) / 2)

        position = np.array(
            [x,y,1]
        )

        dot_x, dot_y, _ = np.dot(matrix, position)
        dot_x = int(dot_x)
        dot_y = int(dot_y)

        #cv2.putText(field_img, name, (dot_x, dot_y + 6), font, 1, (255, 255, 255), 1)
        cv2.circle(field_img, (dot_x,dot_y), 5, (0,0,255), -1)

    
    img_vertical = np.vstack((img, field_img))
    cv2.imshow("test", img_vertical)
    return img_vertical

def main(config):
    field_path = 'C:/Users/kangsanha/Desktop/zappro/zzappro/img/field.png'
    field_img = cv2.imread(field_path)
    field_copy = cv2.imread(field_path)

    video_path = config.video_path
    min_confidence = 0.5
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = load_model()
    model.to(device)

    mot_tracker = Sort()

    cap = cv2.VideoCapture(video_path)

    #재생할 파일의 넓이와 높이
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.4)
    field_copy = cv2.resize(field_copy, (width, height))
    
    frame_size = [width, height]

    print("재생할 파일 넓이, 높이 : %d, %d"%(width, height))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output_test.avi', fourcc, 30.0, (width, height+height))

    start = 0
    if not cap.isOpened:
        print('--@@@@@ ERROR @@@@@--')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--@@@@@ No captured frame @@@@@--')
            break
        else:
            if start == 0:
                point_list = MouseEvent(frame, field_copy) # [point_gt, point_field]
                start += 1
            else:
                img = detection(model, mot_tracker, min_confidence, frame, field_img, point_list, frame_size)
                
                out.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    config = define_argparser()
    main(config)