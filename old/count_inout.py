import cv2
from supervision import BoxAnnotator, Detections, LineZone, LineZoneAnnotator, Point
from ultralytics import YOLO

m = YOLO('yolov8n.pt')
conf = 0.01

file = 'road.mp4'
vcap = cv2.VideoCapture(file)
height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)

line_counter = LineZone(
    start=Point(0, height / 2),
    end=Point(width, height / 2),
)
line_annotator = LineZoneAnnotator(text_scale=1)
box_annotator = BoxAnnotator(text_scale=1)

for i in m.track(source=file, conf=conf, stream=True):
    detections = Detections.from_yolov8(i)
    if i.boxes.id is not None:
        detections.tracker_id = i.boxes.id.cpu().numpy().astype(int)
    line_counter.trigger(detections=detections)
    frame = i.orig_img
    line_annotator.annotate(
        frame=frame,
        line_counter=line_counter,
    )
    labels = [
        f'{conf:0.2f} {m.model.names[cls]} {tracker_id}'
        for _, _, conf, cls, tracker_id in detections
    ]
    frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels,
    )

    cv2.imshow('', frame)

    if cv2.waitKey(1) == 27:
        break
