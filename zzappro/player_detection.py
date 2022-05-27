import cv2 
import time
import numpy as np
import argparse

def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--video_path')
    p.add_argument('--model_file')
    p.add_argument('--cfg_file')
    p.add_argument('--min_confidence', default=0.5)

    config = p.parse_args()

    return config

def detection(net, output_layer, min_confidence, frame, classes):
    start_time = time.time()
    
    img = cv2.resize(frame, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape

    blob_img = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob_img)
    outs = net.forward(output_layer)
    
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            
            scores = detection[5:]
            
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > min_confidence:
                
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
               
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_DUPLEX

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}".format(classes[class_ids[i]])
            
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(img, label, (x, y - 5), font, 1, (255, 255, 255), 1)
    end_time = time.time()
    process_time = end_time - start_time
    print("=== {:.3f} seconds".format(process_time))
    cv2.imshow("test", img)

def main(config):
    video_path = config.video_path

    model_file = config.model_file
    cfg_file = config.cfg_file

    min_confidence = config.min_confidence

    net = cv2.dnn.readNet(model_file, cfg_file)
    
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    classes = []
    with open("./coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    

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
            detection(net, output_layers, min_confidence, frame, classes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    config = define_argparser()
    main(config)