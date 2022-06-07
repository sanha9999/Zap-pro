import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment as linear_assignment

def iou(bb_test, bb_gt):
    """
    [x1, y1, x2, y2] 박스 2개 비교
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h # 
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
        + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o

def convert_bbox_to_z(bbox):
    """
    [x1, y1, x2, y2] 박스를 z로 변환
    [x, y, s, r]의 박스
    x, y : 박스의 중심, s : 면적, r : 가로 세로 비율 
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h 
    r = w / float(h) 
    return np.array([x,y,s,r]).reshape((4,1))

def convert_z_to_bbox(z,score=None):
    """
    [x, y, s, r]의 박스를 [x1, y1, x2, y2]로 변환
    x1, y1 : 왼쪽 상단, x2, y2 : 우측 하단
    """
    w = np.sqrt(z[2] * z[3])
    h = z[2] / w
    if(score==None):
        return np.array([z[0] - w / 2., z[1] - h / 2., z[0] + w / 2., z[1] + h / 2.]).reshape((1,4))
    else:
        return np.array([z[0] - w / 2., z[1] - h / 2., z[0] + w / 2., z[1] + h / 2., score]).reshape((1,5))

class KalmanBoxTracker(object):
    """
    bbox로 tracking object의 내부 상태를 나타낸다.
    """

    count = 0
    def __init__(self, bbox):
        """
        tracker 초기화
        """

        # dim_x : 상태 변수의 수
        # dim_z : 측정 입력의 수(좌표의 수)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # 상태전이행렬
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        
        # 측정 기능
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        # 측정 잡음행렬
        self.kf.R[2:,2:] *= 10.

        # 공분산 행렬
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.

        # 프로세스 잡음행렬
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        # 상태 측정 벡터
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self,bbox):
        """
        상태 업데이트(재귀적)
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        벡터상태수정 and 예측된 bounding box 추정치 반환
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
          self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
          self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_z_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        현재 bounding box 추정치 반환
        """
        return convert_z_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  tracking object(둘 다 bounding box)에 detection을 지정합니다.
  unmatched_detections 및 unmatched_trackers와 match 3 개의 목록을 반환합니다.
  """

  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
  def __init__(self,max_age=1, min_hits=3, iou_threshold=0.3):
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self,dets):
    """
    Params:
      dets - [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    요구 사항 : 빈 frame이있는 경우에도 각 frame마다 한 번 호출해야합니다.
    마지막 열이 object ID 인 유사도 배열을 반환합니다.
    NOTE: 반환 된 object 수는 제공된 detection 수와 다를 수 있습니다.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i][:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))