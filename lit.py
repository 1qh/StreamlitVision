import json
import os
from shutil import which
from subprocess import check_output
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
    Color,
    ColorPalette,
    Detections,
    LineZone,
    LineZoneAnnotator,
    MaskAnnotator,
    Point,
    PolygonZone,
    PolygonZoneAnnotator,
    VideoInfo,
)
from ultralytics import YOLO


def st_config():
    set_page_config(layout='wide')
    st.markdown(
        """
    <style>
    footer {visibility: hidden;}
    @font-face {font-family: 'SF Pro Display';}
    html, body, [class*="css"]  {font-family: 'SF Pro Display';}
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


def custom_params():
    col1, col2 = sb.columns(2)
    with col1:
        model_size = st.selectbox('Model size', ('n', 's', 'm', 'l', 'x'))
    with col2:
        model_map = {
            'Detect': f'{model_size}',
            'Segment': f'{model_size}-seg',
            'Pose': f'{model_size}-pose',
            'Classify': f'{model_size}-cls',
        }
        select = st.selectbox('Model', model_map.keys())
    model = YOLO(f'yolov8{model_map[select]}.pt')
    task = model.overrides['task']
    conf = sb.slider('Threshold', max_value=1.0, value=0.25)
    classes = custom_classes(model)
    return model, task, conf, classes


def hms(s):
    return strftime('%H:%M:%S', gmtime(s))


def plur(n, s):
    return f"\n- {n} {s}{'s'[:n^1]}" if n else ''


def cvt(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def plot(i):
    return cvt(i.plot())


def one_img(file, model):
    st.image(plot(model(Image.open(file))[0]))


def trim_vid(file, path, begin, end):
    trim = f'trim_{file.name}'
    os.system(f'ffmpeg -y -i {path} -ss {begin} -to {end} -c copy {trim}')
    return trim


def first_frame(path):
    cap = cv2.VideoCapture(path)
    frame = Image.fromarray(cvt(cap.read()[1]))
    cap.release()
    return frame


def prepare(file):
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


def draw_tool(model, task, width, height, bg):
    mode = sb.selectbox('Draw', ('line', 'rect', 'polygon'))

    if sb.checkbox('Background', value=True):
        canvas = mycanvas('#000', height, width, mode, bg, key='e')
    else:
        canvas = mycanvas('#fff', height, width, mode, None, key='f')

    lines = []
    polygons = []

    if canvas.json_data is not None:
        draw = canvas.json_data['objects']
        lines, polygons = get_lines_polygons(draw)
        sb.markdown(f"{plur(len(lines), 'line')}{plur(len(polygons), 'polygon')}")

    thickness = sb.slider('Thickness', 0, 10, 1)
    text_scale = sb.slider('Text size', 0.0, 2.0, 0.5)
    text_offset = sb.slider('Text offset', 0.0, 10.0, 1.0)
    text_padding = sb.slider('Text padding', 0, 10, 2)
    text_color = sb.color_picker('Text color', '#000000')

    config = {}
    config['model'] = model.ckpt_path
    config['visual'] = {}

    for i in ('thickness', 'text_scale', 'text_offset', 'text_padding', 'text_color'):
        config['visual'][i] = locals()[i]

    text_color = Color.from_hex(text_color)

    color = ColorPalette.default()

    line_annotator = LineZoneAnnotator(
        thickness=thickness,
        text_color=text_color,
        text_scale=text_scale,
        text_offset=text_offset,
        text_padding=text_padding,
    )
    zones = [
        PolygonZone(
            polygon=p,
            frame_resolution_wh=(width, height),
        )
        for p in polygons
    ]
    zone_annotators = [
        PolygonZoneAnnotator(
            thickness=thickness,
            text_color=text_color,
            text_scale=text_scale,
            text_padding=text_padding,
            zone=z,
            color=color.by_idx(i),
        )
        for i, z in enumerate(zones)
    ]
    box, mask, mask_opacity = None, None, None
    col1, col2 = sb.columns(2)
    with col1:
        use_box = st.checkbox('Box', value=True)
        if use_box:
            box = BoxAnnotator(
                thickness=thickness,
                text_color=text_color,
                text_scale=text_scale,
                text_padding=text_padding,
            )
        config['box'] = use_box
    with col2:
        if task == 'segment':
            use_mask = st.checkbox('Mask', value=True)
            if use_mask:
                mask = MaskAnnotator()
                mask_opacity = sb.slider('Opacity', 0.0, 1.0, 0.5)
                config['mask_opacity'] = mask_opacity
            config['mask'] = use_mask
    config['lines'] = [
        ((l.vector.start.x, l.vector.start.y), (l.vector.end.x, l.vector.end.y))
        for l in lines
    ]
    config['polygons'] = [p.tolist() for p in polygons]

    with open('config.json', 'w') as f:
        json.dump(config, f)

    return lines, line_annotator, zones, zone_annotators, box, mask, mask_opacity


def annot(
    model, res, lines, line_annotator, zones, zone_annotators, box, mask, mask_opacity
):
    det = Detections.from_yolov8(res)
    if res.boxes.id is not None:
        det.tracker_id = res.boxes.id.cpu().numpy().astype(int)
    if res.masks is not None:
        det.mask = res.masks.data
    f = res.orig_img

    if box:
        f = box.annotate(
            scene=f,
            detections=det,
            labels=[
                f'{conf:0.2f} {model.model.names[cls]}'
                + (f' {track_id}' if track_id else '')
                for _, _, conf, cls, track_id in det
            ],
        )
    if mask:
        f = mask.annotate(
            scene=f,
            detections=det,
            opacity=mask_opacity,
        )
    for l in lines:
        l.trigger(detections=det)
        line_annotator.annotate(frame=f, line_counter=l)

    for z, zone in zip(zones, zone_annotators):
        det.mask = None  # bug in supervision
        z.trigger(detections=det)
        f = zone.annotate(scene=f)
    return cvt(f)


def app(state):
    st_config()
    m, task, conf, classes = custom_params()

    def model(
        source,
        classes=classes,
        conf=conf,
        stream=False,
        track=False,
    ):
        if track:
            return m.track(
                source, classes=classes, conf=conf, retina_masks=True, stream=stream
            )
        return m(source, classes=classes, conf=conf, retina_masks=True, stream=stream)

    def cam(frame):
        f = plot(model(frame.to_ndarray(format='bgr24'))[0])
        # print(f.shape)
        # oh my god, it took me so long to realize this increases through time
        return VideoFrame.from_ndarray(f)

    def cam_track(frame):
        f = plot(model(frame.to_ndarray(format='bgr24'), track=True)[0])
        return VideoFrame.from_ndarray(f)

    if sb.checkbox('Use Camera'):
        track = False
        reso = check_output(
            "v4l2-ctl -d /dev/video0 --list-formats-ext | grep Size: | tail -1 | awk '{print $NF}'",
            shell=True,
        )
        width, height = [int(i) for i in reso.decode().split('x')]
        if task != 'classify':
            track = sb.checkbox('Track')

        if track:
            webrtc_streamer(key='a', video_frame_callback=cam_track)
        else:
            webrtc_streamer(key='b', video_frame_callback=cam)

        if sb.button('Native Run'):
            os.system(f'./native.py --path 0')

        picture = st.camera_input('Shoot')
        if picture:
            one_img(picture, model)
            bg = Image.open(picture).resize((width, height))
            (
                lines,
                line_annotator,
                zones,
                zone_annotators,
                box,
                mask,
                mask_opacity,
            ) = draw_tool(
                m,
                task,
                bg.size[0],
                bg.size[1],
                bg,
            )

    file = sb.file_uploader(' ')
    if file:
        if 'image' in file.type:
            sb.image(file)
            one_img(file, model)
        elif 'video' in file.type:
            vid, path, trimmed, begin, end = prepare(file)
            width, height = vid.resolution_wh
            track = False
            if task != 'classify':
                track = sb.checkbox('Track')
                (
                    lines,
                    line_annotator,
                    zones,
                    zone_annotators,
                    box,
                    mask,
                    mask_opacity,
                ) = draw_tool(
                    m,
                    task,
                    width,
                    height,
                    first_frame(path),
                )
            sv_out = None

            if sb.button('Native Run'):
                os.system(f'./native.py --path {path}')

            while sb.checkbox('Run', key='r'):
                if trimmed:
                    path = trim_vid(file, path, begin, end)

                with st.empty():
                    for res in model(path, stream=True, track=track):
                        yolo_out = plot(res)
                        sv_out = (
                            annot(
                                m,
                                res,
                                lines,
                                line_annotator,
                                zones,
                                zone_annotators,
                                box,
                                mask,
                                mask_opacity,
                            )
                            if task != 'classify'
                            else yolo_out
                        )
                        tab1, tab2 = st.tabs(['Supervision', 'YOLO'])
                        with tab1:
                            st.image(sv_out)
                        with tab2:
                            st.image(yolo_out)
        else:
            sb.warning('Please upload image/video')


class YoloFlow(LightningFlow):
    def configure_layout(self):
        return StreamlitFrontend(render_fn=app)

    def run(self):
        pass


lit = LightningApp(YoloFlow())

app('')  # comment this line to run in lightning AI
