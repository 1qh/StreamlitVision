import os
from shutil import which
from time import gmtime, strftime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit import set_page_config
from streamlit import sidebar as sb
from streamlit_drawable_canvas import st_canvas
from supervision import (
    BoxAnnotator,
    ColorPalette,
    Detections,
    LineZone,
    LineZoneAnnotator,
    Point,
    PolygonZone,
    PolygonZoneAnnotator,
    VideoInfo,
)
from ultralytics import YOLO

set_page_config(layout='wide')


def custom_classes(model):
    d = model.names
    all = list(d.values())

    if sb.checkbox('Custom Classes'):
        return [all.index(i) for i in sb.multiselect(' ', all)]
    else:
        return list(d.keys())


def hms(s):
    return strftime('%H:%M:%S', gmtime(s))


def plur(n, s):
    if n:
        return f"\n- {n} {s}{'s'[:n^1]}"
    return ''


def first_frame(path):
    vcap = cv2.VideoCapture(path)
    frame = Image.fromarray(cv2.cvtColor(vcap.read()[1], cv2.COLOR_BGR2RGB))
    vcap.release()
    return frame


conf = sb.slider('Threshold', max_value=1.0, value=0.25)
model = YOLO('yolov8n.pt')
classes = custom_classes(model)
file = sb.file_uploader(' ')

if file and 'video' in file.type:
    sb.video(file)
    path = f'up_{file.name}'

    with open(path, 'wb') as up:
        up.write(file.read())

    vid = VideoInfo.from_video_path(path)
    width, height = vid.resolution_wh
    mode = sb.selectbox('Draw', ('line', 'rect', 'polygon', 'transform'))
    text_scale = sb.slider('Text size', 0.0, 2.0, 1.0)
    canvas = st_canvas(
        stroke_width=2,
        fill_color='#ffffff55',
        drawing_mode=mode,
        background_image=first_frame(path),
        height=height,
        width=width,
        key='a',
    )
    polygons = []

    if canvas.json_data is not None:
        draw = canvas.json_data['objects']
        lines = [
            LineZone(
                start=Point(i['left'] + i['x1'], i['top'] + i['y1']),
                end=Point(i['left'] + i['x2'], i['top'] + i['y2']),
            )
            for i in draw
            if i['type'] == 'line'
        ]
        polygons = [
            np.array([[x[1], x[2]] for x in k], np.int32)
            for k in [j[:-1] for j in [i['path'] for i in draw if i['type'] == 'path']]
        ] + [
            np.array(
                [
                    [i['left'], i['top']],
                    [i['left'] + i['width'], i['top']],
                    [i['left'] + i['width'], i['top'] + i['height']],
                    [i['left'], i['top'] + i['height']],
                ],
                np.int32,
            )
            for i in draw
            if i['type'] == 'rect'
        ]
        sb.markdown(f"{plur(len(lines), 'line')}{plur(len(polygons), 'polygon')}")

    line = LineZoneAnnotator(text_scale=text_scale)
    box = BoxAnnotator(text_scale=text_scale)
    colors = ColorPalette.default()
    zones = [
        PolygonZone(polygon=p, frame_resolution_wh=vid.resolution_wh) for p in polygons
    ]
    zone_annotators = [
        PolygonZoneAnnotator(text_scale=text_scale, zone=z, color=colors.by_idx(i))
        for i, z in enumerate(zones)
    ]
    box_annotators = [
        BoxAnnotator(text_scale=text_scale, color=colors.by_idx(i))
        for i in range(len(polygons))
    ]
    if which('ffmpeg'):
        trimmed = sb.checkbox('Trim')

        if trimmed:
            length = int(vid.total_frames / vid.fps)
            begin, end = sb.slider(
                'Trim by second',
                value=(
                    0,
                    length,
                ),
                max_value=length,
            )
            begin, end = hms(begin), hms(end)
            sb.write(f'Trim from {begin} to {end}')
            trim = f'trim_{file.name}'

    while sb.checkbox('Run', key='r'):
        if trimmed:
            os.system(f'ffmpeg -y -i {path} -ss {begin} -to {end} -c copy {trim}')
            path = trim

        with st.empty():
            for i in model.track(
                path,
                stream=True,
                classes=classes,
                conf=conf,
                retina_masks=True,
            ):
                det = Detections.from_yolov8(i)
                if i.boxes.id is not None:
                    det.tracker_id = i.boxes.id.cpu().numpy().astype(int)
                f = i.orig_img

                for z, zone, box in zip(zones, zone_annotators, box_annotators):
                    z.trigger(detections=det)
                    f = box.annotate(scene=f, detections=det)
                    f = zone.annotate(scene=f)

                for l in lines:
                    l.trigger(detections=det)
                    line.annotate(frame=f, line_counter=l)

                f = box.annotate(
                    scene=f,
                    detections=det,
                    labels=[
                        f'{conf:0.2f} {model.model.names[cls]} {tracker_id}'
                        for _, conf, cls, tracker_id in det
                    ],
                )
                st.image(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
else:
    sb.warning('Please upload video')
