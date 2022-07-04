import cv2
import numpy as np 

def mouse_handler(event, x, y, flags, data):

    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y), 3, (0,0,255), -1)
        cv2.imshow('point', data['im'])

        if len(data['points']) < 4 :
            data['points'].append([x,y])

def get_background_points(img):
    data = {}
    data['im'] = img.copy()
    data['points'] = []

    cv2.imshow('point', img)
    cv2.setMouseCallback('point', mouse_handler, data)
    cv2.waitKey()
    cv2.destroyAllWindows()

    #float
    points = np.array(data['points'], dtype=float)

    return points

def MouseEvent(frame, field_img):
    
    frame = cv2.resize(frame, None, fx=0.8, fy=0.8)
    point_gt = get_background_points(frame)
    
    point_field = get_background_points(field_img)
    
    return [point_gt, point_field]
