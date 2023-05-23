import json
import os
from pathlib import Path
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
from psutil import process_iter
from streamlit import session_state, set_page_config
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
    draw_text,
    get_polygon_center,
)
from ultralytics import YOLO

from color import colors, colors_rgb
from native import annot

_shape = None
_lines, _zones, _zone_ann = [], [], []

if 'path' not in session_state:
    session_state['path'] = ''


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
        return [
            all.index(i) for i in sb.multiselect(' ', all, label_visibility='collapsed')
        ]
    else:
        return list(d.keys())


def custom_params(config):
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
    conf = sb.slider('Threshold', max_value=1.0, value=0.25)
    classes = custom_classes(model)

    config['model'] = model.ckpt_path
    config['conf'] = conf
    config['classes'] = classes

    return config, model, conf, classes


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


def rgb2hex(rgb):
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'


def trim_vid(path, begin, end):
    trim = f'trim_{path[3:]}'
    os.system(f'ffmpeg -y -i {path} -ss {begin} -to {end} -c copy {trim}')
    return trim


def first_frame(path):
    cap = cv2.VideoCapture(path)
    frame = Image.fromarray(cvt(cap.read()[1]))
    cap.release()
    return frame


def prepare(path):
    vid = VideoInfo.from_video_path(path)

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
            if sb.button(f'Trim from {begin[3:]} to {end[3:]}'):
                path = trim_vid(path, begin, end)
                session_state['path'] = path
        else:
            session_state['path'] = path
    else:
        session_state['path'] = path


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


def draw_tool(config, task, width, height, bg):
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

    config['lines'] = [
        ((l.vector.start.x, l.vector.start.y), (l.vector.end.x, l.vector.end.y))
        for l in lines
    ]
    config['polygons'] = [p.tolist() for p in polygons]

    thickness = sb.slider('Thickness', 0, 10, 1)
    text_scale = sb.slider('Text size', 0.0, 2.0, 0.5)
    text_offset = sb.slider('Text offset', 0.0, 10.0, 1.0)
    text_padding = sb.slider('Text padding', 0, 10, 2)
    text_color = sb.color_picker('Text color', '#000000')

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
    box, mask, mask_opacity, area = None, None, None, None
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
                area = sb.checkbox('Mask area', value=True)
                config['area'] = area
                mask_opacity = sb.slider('Opacity', 0.0, 1.0, 0.5)
                config['mask_opacity'] = mask_opacity
            config['mask'] = use_mask
    predict_color = sb.checkbox('Predict color', value=False)
    if predict_color:
        for color, rgb in zip(colors, colors_rgb):
            sb.color_picker(f'{color}', value=rgb2hex(rgb))
    config['predict_color'] = predict_color
    return (
        config,
        lines,
        line_annotator,
        zones,
        zone_annotators,
        box,
        mask,
        mask_opacity,
        area,
        predict_color,
    )


def save_config(config):
    if sb.button('Save config'):
        with open('config.json', 'w') as f:
            json.dump(config, f)


def native_run(source):
    here = Path(__file__).parent
    option = sb.radio(f'Native run on {source}', ('Show', 'Save to video'))
    if option == 'Show':
        if sb.button('Show with OpenCV'):
            cmd = f'{here}/native.py --source {source}'
            st.code(cmd, language='bash')
            os.system(cmd)
    elif option == 'Save to video':
        saveto = sb.text_input(' ', 'result.mp4', label_visibility='collapsed')
        if sb.button('Save with OpenCV'):
            cmd = f'{here}/native.py --source {source} --saveto {saveto}'
            st.code(cmd, language='bash')
            os.system(cmd)


def update_annot(f, height, lines, zones, _zone_ann):
    _shape = f.shape
    scale = _shape[0] / height
    _lines = [
        LineZone(
            start=Point(
                l.vector.start.x * scale,
                l.vector.start.y * scale,
            ),
            end=Point(
                l.vector.end.x * scale,
                l.vector.end.y * scale,
            ),
        )
        for l in lines
    ]
    _zones = [
        PolygonZone(
            polygon=(z.polygon * scale).astype(int),
            frame_resolution_wh=(_shape[1], _shape[0]),
        )
        for z in zones
    ]
    for z, ann in zip(_zones, _zone_ann):
        ann.zone = z
        ann.center = get_polygon_center(polygon=z.polygon)

    return _lines, _zones, _zone_ann, _shape


def webui(state):
    st_config()
    config = {}
    config, m, conf, classes = custom_params(config)
    allclasses = m.model.names
    task = m.overrides['task']

    def model(
        source,
        classes=classes,
        conf=conf,
        stream=False,
        tracker=None,
    ):
        if tracker is not None:
            return m.track(
                source,
                classes=classes,
                conf=conf,
                retina_masks=True,
                stream=stream,
                tracker=f'{tracker}.yaml',
            )
        return m(source, classes=classes, conf=conf, retina_masks=True, stream=stream)

    file = sb.file_uploader(' ', label_visibility='collapsed')
    use_cam = sb.checkbox('Use Camera', value=True if not file else False)

    if use_cam:
        file = None
        reso = check_output(
            "v4l2-ctl -d /dev/video0 --list-formats-ext | grep Size: | tail -1 | awk '{print $NF}'",
            shell=True,
        )
        width, height = [int(i) for i in reso.decode().split('x')]

        def cam_stream(key, callback):
            webrtc_streamer(
                key=key,
                video_frame_callback=callback,
                media_stream_constraints={
                    'video': {
                        'width': {'min': width},
                        'height': {'min': height},
                    }
                },
            )

        tracker = None
        if task != 'classify':
            tracker = sb.selectbox('Tracker', [None, 'bytetrack', 'botsort'])
        else:
            tracker = False
        config['tracker'] = tracker

        def cam(frame):
            return VideoFrame.from_ndarray(
                model(frame.to_ndarray(), tracker=tracker)[0].plot()
            )

        cam_stream('a', cam)

        picture = None
        if sb.checkbox('Annotate from selfie'):
            picture = st.camera_input('Shoot')

        if picture:
            one_img(picture, model)
            bg = Image.open(picture).resize((width, height))
            (
                config,
                lines,
                line_annotator,
                zones,
                zone_annotators,
                box,
                mask,
                mask_opacity,
                area,
                predict_color,
            ) = draw_tool(
                config,
                task,
                width,
                height,
                bg,
            )
            save_config(config)
            native_run(0)
            cam_open = sb.checkbox('Run')
            cap = cv2.VideoCapture(0)
            codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            cap.set(6, codec)
            cap.set(5, 30)
            cap.set(3, width)
            cap.set(4, height)
            with st.empty():
                while cam_open:
                    _, f = cap.read()
                    res = model(f, tracker=tracker)[0]
                    frame = annot(
                        allclasses,
                        res,
                        lines,
                        line_annotator,
                        zones,
                        zone_annotators,
                        box,
                        mask,
                        mask_opacity,
                        area,
                        predict_color,
                    )
                    st.image(cvt(frame))
            cap.release()

            # oh my god, it took me so long to realize the frame bigger through time
            def cam_annot(frame):
                f = frame.to_ndarray()
                global _shape, _lines, _zones, _zone_ann
                _zone_ann = zone_annotators
                if f.shape != _shape:
                    _lines, _zones, _zone_ann, _shape = update_annot(
                        f, height, lines, zones, _zone_ann
                    )
                f = annot(
                    allclasses,
                    model(f, tracker=tracker)[0],
                    _lines,
                    line_annotator,
                    _zones,
                    _zone_ann,
                    box,
                    mask,
                    mask_opacity,
                    area,
                    predict_color,
                )
                return VideoFrame.from_ndarray(f)

            cam_stream('b', cam_annot)

    if file:
        if 'image' in file.type:
            sb.image(file)
            one_img(file, model)

        elif 'video' in file.type:
            sb.video(file)
            path = f'up_{file.name}'
            with open(path, 'wb') as up:
                up.write(file.read())

            prepare(path)
            path = session_state['path']
            vid = VideoInfo.from_video_path(path)

            width, height = vid.resolution_wh
            sb.markdown(
                f'''
                - Video resolution: {width}x{height}
                - Total frames: {vid.total_frames}
                - FPS: {vid.fps}
                - Path: {path}
                '''
            )

            tracker = None
            if task != 'classify':
                tracker = sb.selectbox('Tracker', [None, 'bytetrack', 'botsort'])
                (
                    config,
                    lines,
                    line_annotator,
                    zones,
                    zone_annotators,
                    box,
                    mask,
                    mask_opacity,
                    area,
                    predict_color,
                ) = draw_tool(
                    config,
                    task,
                    width,
                    height,
                    first_frame(path),
                )
            else:
                tracker = None
            config['tracker'] = tracker

            sv_out = None
            run = sb.checkbox('Run & show on web', key='r')
            with st.empty():
                while run:
                    for res in model(path, stream=True, tracker=tracker):
                        yolo_out = plot(res)
                        sv_out = (
                            annot(
                                allclasses,
                                res,
                                lines,
                                line_annotator,
                                zones,
                                zone_annotators,
                                box,
                                mask,
                                mask_opacity,
                                area,
                                predict_color,
                            )
                            if task != 'classify'
                            else yolo_out
                        )
                        tab1, tab2 = st.tabs(['Supervision', 'YOLO'])
                        with tab1:
                            st.image(cvt(sv_out))
                        with tab2:
                            st.image(yolo_out)
            save_config(config)
            native_run(path)
        else:
            sb.warning('Please upload image/video')


class App(LightningFlow):
    def configure_layout(self):
        return StreamlitFrontend(render_fn=webui)

    def run(self):
        pass


lit = LightningApp(App())

running_apps = [i for i in [p.cmdline() for p in process_iter()] if 'run' in i]
this_process = next(p for p in running_apps if any(Path(__file__).stem in a for a in p))

if 'app' not in this_process:
    webui('')
