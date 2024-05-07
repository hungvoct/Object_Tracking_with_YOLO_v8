from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet
import numpy as np
import cv2
from detection import YOLOv8
import os



class DeepSORT:
    def __init__(self, model_path='/home/kehungvo/object_tracking/resources/networks/mars-small128.pb', max_cosine_distance=0.7, nn_budget=None, classes=['objects']):
        self.encoder = gdet.create_box_encoder(model_path, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metric)

        key_list = []
        val_list = []
        for ID, class_name in enumerate(classes):
            key_list.append(ID)
            val_list.append(class_name)
        self.key_list = key_list
        self.val_list = val_list

    def tracking(self, origin_frame, bboxes, scores, class_ids):
        features = self.encoder(origin_frame, bboxes)

        detections = [Detection(bbox, score, class_id, feature)
                      for bbox, score, class_id, feature in zip(bboxes, scores, class_ids, features)]

        self.tracker.predict()
        self.tracker.update(detections)

        tracked_bboxes = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
            bbox = track.to_tlbr()
            class_id = track.get_class()
            conf_score = track.get_conf_score()
            tracking_id = track.track_id
            tracked_bboxes.append(bbox.tolist() + [class_id, conf_score, tracking_id])

        tracked_bboxes = np.array(tracked_bboxes)

        return tracked_bboxes

def draw_detection(img, bboxes, scores, class_ids, ids, classes=['objects'], mask_alpha=0.3):
    height, width = img.shape[:2]
    np.random.seed(0)
    rng = np.random.default_rng(3)
    colors = rng.uniform(0, 255, size=(len(classes), 3))

    mask_img = img.copy()
    det_img = img.copy()

    size = min([height, width]) * 0.0006
    text_thickness = int(min([height, width]) * 0.001)

    # Vẽ các hộp giới hạn và nhãn của các đối tượng được phát hiện
    for bbox, score, class_id, id_ in zip(bboxes, scores, class_ids, ids):
        color = colors[class_id]

        x1, y1, x2, y2 = bbox.astype(int)

        # Vẽ hình chữ nhật đầy màu trên hình ảnh mặt nạ
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        label = classes[class_id]
        caption = f'{label} {int(score * 100)}% ID: {id_}'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        # Vẽ hình chữ nhật đầy màu và nhãn trên hình ảnh chứa các hộp giới hạn
        cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
        cv2.putText(det_img, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(mask_img, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    # Kết hợp hình ảnh mặt nạ và hình ảnh chứa các hộp giới hạn
    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

def video_tracking( detector, tracker):
    # Mở video từ đường dẫn được cung cấp
    cap = cv2.VideoCapture('/home/kehungvo/object_tracking/video_mu.webm')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



    all_tracking_results = []
    tracked_ids = np.array([], dtype=np.int32)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Phát hiện đối tượng trên khung hình
        detector_results = detector.detect(frame)
        bboxes, scores, class_ids = detector_results

        # Dự đoán đối tượng sử dụng trình theo dõi
        tracker_pred = tracker.tracking(origin_frame=frame, bboxes=bboxes, scores=scores, class_ids=class_ids)

        if tracker_pred.size > 0:
            # Lấy các thông tin từ dự đoán của trình theo dõi
            bboxes = tracker_pred[:, :4]
            class_ids = tracker_pred[:, 4].astype(int)
            conf_scores = tracker_pred[:, 5]
            tracking_ids = tracker_pred[:, 6].astype(int)

            # Lấy các ID theo dõi mới
            new_ids = np.setdiff1d(tracking_ids, tracked_ids)

            # Lưu trữ các ID theo dõi mới
            tracked_ids = np.concatenate((tracked_ids, new_ids))

            # Vẽ các hộp giới hạn và nhãn trên khung hình
            result_img = draw_detection(img=frame, bboxes=bboxes, scores=conf_scores, class_ids=class_ids, ids=tracking_ids)
        else:
            result_img = frame

        all_tracking_results.append(tracker_pred)

        cv2.imshow('result',result_img)

        # Thoát khỏi vòng lặp nếu phím 'q' được nhấn
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    cap.release()


    cv2.destroyAllWindows()

    return all_tracking_results

detector = YOLOv8('/home/kehungvo/object_tracking/yolov8n.pt')
model_track = DeepSORT()

all_tracking_results = video_tracking(
    detector,
    model_track)

