import os
from shutil import which
from time import gmtime, strftime

import cv2
import numpy as np
import streamlit as st
from av import VideoFrame
from lightning import LightningApp, LightningFlow
from lightning.app.frontend import StreamlitFrontend
from PIL import Image
from streamlit import set_page_config
from streamlit import sidebar as sb
from streamlit_drawable_canvas import st_canvas
from streamlit_webrtc import webrtc_streamer
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

model_map = {
    'Detect': 'n',
    'Segment': 'n-seg',
    'Pose': 'n-pose',
    'Classify': 'n-cls',
}

### UTILS ###


def st_config():
    set_page_config(layout='wide')
    st.markdown(
        """
    <style>
    footer {visibility: hidden;}
    # @font-face {font-family: 'SF Pro Display';}
    # html, body, [class*="css"]  {font-family: 'SF Pro Display';}
    thead tr th:first-child {display:none}
    tbody th {display:none}
    </style>
    """,
        unsafe_allow_html=True,
    )


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


def custom_params(model_map):
    select = sb.selectbox('Model', model_map.keys())
    model = YOLO(f'yolov8{model_map[select]}.pt')
    conf = sb.slider('Threshold', max_value=1.0, value=0.25)
    classes = custom_classes(model)
    return model, conf, classes


def prepare(file):
    if 'video' in file.type:
        sb.video(file)
        path = f'up_{file.name}'
        with open(path, 'wb') as up:
            up.write(file.read())
        vid = VideoInfo.from_video_path(path)

        trimmed, begin, end = False, None, None
        if which('ffmpeg'):
            trimmed = sb.checkbox('Trim')
            if trimmed:
                length = int(vid.total_frames / vid.fps)
                begin, end = sb.slider(
                    'Trim by second',
                    value=(0, length),
                    max_value=length,
                )
                begin, end = hms(begin), hms(end)
                sb.write(f'Trim from {begin} to {end}')
        return vid, path, trimmed, begin, end


def trim_vid(file, path, begin, end):
    trim = f'trim_{file.name}'
    os.system(f'ffmpeg -y -i {path} -ss {begin} -to {end} -c copy {trim}')
    return trim


### HOME APP ###


def get_frame(i):
    return cv2.cvtColor(i.plot(), cv2.COLOR_BGR2RGB)


def from_model(s, model):
    return get_frame(model(s)[0])


def infer_image(file, model):
    st.image(from_model(Image.open(file), model))


def det_app(_):
    st_config()

    model, conf, classes = custom_params(model_map)

    def mymodel(source, stream=False):
        return model(
            source, classes=classes, conf=conf, retina_masks=True, stream=stream
        )

    def cam_stream(frame):
        return VideoFrame.from_ndarray(
            from_model(frame.to_ndarray(format='bgr24'), mymodel)
        )

    if sb.checkbox('Use Camera'):
        picture = st.camera_input('Shoot')
        if picture:
            infer_image(picture, mymodel)
        webrtc_streamer(key='a', video_frame_callback=cam_stream)

    file = sb.file_uploader(' ')
    if file:
        if 'image' in file.type:
            sb.image(file)
            infer_image(file, mymodel)
        else:
            vid, path, trimmed, begin, end = prepare(file)

            while sb.checkbox('Run'):
                if trimmed:
                    path = trim_vid(file, path, begin, end)

                with st.empty():
                    for res in mymodel(path, stream=True):
                        st.image(get_frame(res))


### TRACK APP ###


def get_lines_polygons(d):
    lines = [
        LineZone(
            start=Point(i['left'] + i['x1'], i['top'] + i['y1']),
            end=Point(i['left'] + i['x2'], i['top'] + i['y2']),
        )
        for i in d
        if i['type'] == 'line'
    ]
    polygons = [
        np.array([[x[1], x[2]] for x in k], np.int32)
        for k in [j[:-1] for j in [i['path'] for i in d if i['type'] == 'path']]
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
        for i in d
        if i['type'] == 'rect'
    ]
    return lines, polygons


def first_frame(path):
    vcap = cv2.VideoCapture(path)
    frame = Image.fromarray(cv2.cvtColor(vcap.read()[1], cv2.COLOR_BGR2RGB))
    vcap.release()
    return frame


def mycanvas(stroke, height, width, mode, bg, key):
    return st_canvas(
        stroke_width=2,
        fill_color='#ffffff55',
        stroke_color=stroke,
        drawing_mode=mode,
        background_image=bg,
        height=height,
        width=width,
        key=key,
    )


def annot(model, res, lines, line_annotator, zones, zone_annotators, box):
    det = Detections.from_yolov8(res)
    if res.boxes.id is not None:
        det.tracker_id = res.boxes.id.cpu().numpy().astype(int)
    f = res.orig_img

    for l in lines:
        l.trigger(detections=det)
        line_annotator.annotate(frame=f, line_counter=l)

    for z, zone in zip(zones, zone_annotators):
        z.trigger(detections=det)
        f = zone.annotate(scene=f)

    return cv2.cvtColor(
        box.annotate(
            scene=f,
            detections=det,
            labels=[
                f'{conf:0.2f} {model.model.names[cls]} {tracker_id}'
                for _, _, conf, cls, tracker_id in det
            ],
        ),
        cv2.COLOR_BGR2RGB,
    )


def track_app(_):
    st_config()
    model, conf, classes = custom_params(model_map)

    file = sb.file_uploader(' ')
    if file:
        vid, path, trimmed, begin, end = prepare(file)
        mode = sb.selectbox('Draw', ('line', 'rect', 'polygon'))
        width, height = vid.resolution_wh

        if sb.checkbox('Background', value=True):
            canvas = mycanvas('#000', height, width, mode, first_frame(path), key='a')
        else:
            canvas = mycanvas('#fff', height, width, mode, None, key='b')

        lines = []
        polygons = []

        if canvas.json_data is not None:
            draw = canvas.json_data['objects']
            lines, polygons = get_lines_polygons(draw)
            sb.markdown(f"{plur(len(lines), 'line')}{plur(len(polygons), 'polygon')}")

        text_scale = sb.slider('Text size', 0.0, 2.0, 1.0)
        color = ColorPalette.default()
        line_annotator = LineZoneAnnotator(text_scale=text_scale)
        box = BoxAnnotator(text_scale=text_scale)
        zones = [
            PolygonZone(polygon=p, frame_resolution_wh=vid.resolution_wh)
            for p in polygons
        ]
        zone_annotators = [
            PolygonZoneAnnotator(text_scale=text_scale, zone=z, color=color.by_idx(i))
            for i, z in enumerate(zones)
        ]

        while sb.checkbox('Run', key='r'):
            if trimmed:
                path = trim_vid(file, path, begin, end)

            with st.empty():
                for res in model.track(
                    path,
                    stream=True,
                    classes=classes,
                    conf=conf,
                    retina_masks=True,
                ):
                    tab1, tab2 = st.tabs(['YOLO', 'Supervision'])
                    with tab1:
                        st.image(get_frame(res))
                    with tab2:
                        st.image(
                            annot(
                                model,
                                res,
                                lines,
                                line_annotator,
                                zones,
                                zone_annotators,
                                box,
                            )
                        )
    else:
        sb.warning('Please upload video')


def test(_):
    st.write('test')


### SETUP ###


class Det(LightningFlow):
    def configure_layout(self):
        return StreamlitFrontend(render_fn=det_app)


class Track(LightningFlow):
    def configure_layout(self):
        return StreamlitFrontend(render_fn=track_app)


class Test(LightningFlow):
    def configure_layout(self):
        return StreamlitFrontend(render_fn=test)


class LitApp(LightningFlow):
    def __init__(self):
        super().__init__()
        self.det = Det()
        self.track = Track()
        self.test = Test()

    def run(self):
        self.det.run()
        self.track.run()
        self.test.run()

    def configure_layout(self):
        return [
            {'name': 'home', 'content': self.det},
            {'name': 'track', 'content': self.track},
            {'name': 'test', 'content': self.test},
        ]


app = LightningApp(LitApp())
