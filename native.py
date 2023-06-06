#!/usr/bin/env python3
import argparse
import json
import time
from subprocess import check_output

import cv2
import numpy as np
import yolov5
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
    VideoSink,
    draw_text,
    get_video_frames_generator,
)
from ultralytics import RTDETR, YOLO
from vidgear.gears import VideoGear, WriteGear

from color import colors, colors_rgb


def rgb2ycc(rgb):
    rgb = rgb / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331364 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def closest(rgb, ycc_colors):
    return np.argmin(np.sum((ycc_colors - rgb2ycc(rgb[np.newaxis])) ** 2, axis=1))


def cvt(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def avg_rgb(f):
    return cv2.kmeans(
        cvt(f.reshape(-1, 3).astype(np.float32)),
        1,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )[2][0].astype(np.int32)


ycc_colors = rgb2ycc(colors_rgb)
colors_rgb = [tuple(map(int, i)) for i in colors_rgb]


def get_det(f, model, legacy=False, tracker=None):
    if legacy:
        return Detections.from_yolov5(model(f)), cvt(f)
    res = model(f, tracker=tracker)[0]
    if res.boxes is not None:
        det = Detections.from_yolov8(res)
        if res.boxes.id is not None:
            det.tracker_id = res.boxes.id.cpu().numpy().astype(int)
        if res.masks is not None:
            det.mask = res.masks.data.cpu().numpy()
        return det, cvt(res.plot())
    return Detections.empty(), cvt(res.plot())


def annot(
    f,
    model,
    tracker,
    legacy,
    allclasses,
    lines,
    line_annotator,
    zones,
    zone_annotators,
    box,
    skip_label,
    mask,
    mask_opacity,
    area,
    predict_color,
    show_fps,
):
    begin = time.time()
    det, res = (
        get_det(f, model, legacy) if legacy else get_det(f, model, tracker=tracker)
    )

    xyxy = det.xyxy.astype(int)

    text_color = line_annotator.text_color
    text_scale = line_annotator.text_scale
    text_padding = line_annotator.text_padding

    if predict_color:
        naive = False
        centers = (xyxy[:, [0, 1]] + xyxy[:, [2, 3]]) // 2

        for i in range(xyxy.shape[0]):
            x = centers[i][0]
            y = centers[i][1]

            bb = xyxy[i]

            # for shirt color of person
            w = bb[2] - bb[0]
            h = bb[3] - bb[1]
            crop = f[
                bb[1] : bb[3] - int(h * 0.4),
                bb[0] + int(w * 0.2) : bb[2] - int(w * 0.2),
            ]

            # crop = f[bb[1] : bb[3], bb[0] : bb[2]]
            rgb = f[y, x] if naive else avg_rgb(crop)
            predict = closest(rgb, ycc_colors)
            r, g, b = colors_rgb[predict]
            draw_text(
                scene=f,
                text=colors[predict],
                text_anchor=Point(x=x, y=y + 20),
                text_color=Color(255 - r, 255 - g, 255 - b),
                text_scale=text_scale,
                text_padding=text_padding,
                background_color=Color(r, g, b),
            )
    if box:
        f = box.annotate(
            scene=f,
            detections=det,
            labels=[
                f'{conf:0.2f} {allclasses[cls]}' + (f' {track_id}' if track_id else '')
                for _, _, conf, cls, track_id in det
            ],
            skip_label=skip_label,
        )
    if mask:
        f = mask.annotate(scene=f, detections=det, opacity=mask_opacity)
    if mask and area:
        for t, a in zip(det.area, xyxy.astype(int)):
            draw_text(
                scene=f,
                text=f'{int(t)}',
                text_anchor=Point(x=(a[0] + a[2]) // 2, y=(a[1] + a[3]) // 2),
                text_color=text_color,
                text_scale=text_scale,
                text_padding=text_padding,
            )
    for l in lines:
        l.trigger(detections=det)
        line_annotator.annotate(frame=f, line_counter=l)

    for z, zone in zip(zones, zone_annotators):
        z.trigger(detections=det)
        f = zone.annotate(scene=f)

    fps = 1 / (time.time() - begin)
    if show_fps:
        draw_text(
            scene=f,
            text=f'{fps:.1f}',
            text_anchor=Point(x=50, y=20),
            text_color=text_color,
            text_scale=text_scale * 2,
            text_padding=text_padding,
        )
    return f, res


def maxcam():
    reso = check_output(
        "v4l2-ctl -d /dev/video0 --list-formats-ext | grep Size: | tail -1 | awk '{print $NF}'",
        shell=True,
    )
    width, height = [int(i) for i in reso.decode().split('x')]
    return width, height


def save(source, target, callback):
    if type(source) == int:  # not working yet
        width, height = maxcam()
        with VideoSink(target, VideoInfo(width, height, 30)) as sink:
            for frame in get_video_frames_generator(source):
                sink.write_frame(callback(frame))
    else:
        with VideoSink(target, VideoInfo.from_video_path(source)) as sink:
            for frame in get_video_frames_generator(source):
                sink.write_frame(callback(frame))


def init_annotator(config, reso, polygons):
    use_box = config.get('box', False)
    skip_label = config.get('skip_label', False)
    use_mask = config.get('mask', False)
    area = config.get('area', False)
    predict_color = config.get('predict_color', False)
    show_fps = config.get('show_fps', False)
    visual = config.get('visual', {})

    thickness = 1
    text_scale = 0.5
    text_offset = 1
    text_padding = 2
    text_color = '#000000'

    if visual != {}:
        thickness = visual['thickness']
        text_scale = visual['text_scale']
        text_offset = visual['text_offset']
        text_padding = visual['text_padding']
        text_color = visual['text_color']

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
            frame_resolution_wh=reso,
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

    if use_box:
        box = BoxAnnotator(
            thickness=thickness,
            text_color=text_color,
            text_scale=text_scale,
            text_padding=text_padding,
        )
    if use_mask:
        mask = MaskAnnotator()
        mask_opacity = config['mask_opacity'] if 'mask_opacity' in config else 0.5

    return (
        line_annotator,
        zones,
        zone_annotators,
        box,
        skip_label,
        mask,
        mask_opacity,
        area,
        predict_color,
        show_fps,
    )


def app(source, config='config.json', saveto=None):
    if '.' not in source and int(source) in [i for i in range(-1, 2, 1)]:
        source = int(source)
        reso = maxcam()
        fps = 30
    else:
        vid = VideoInfo.from_video_path(source)
        reso = vid.resolution_wh
        fps = vid.fps

    config = json.load(open(config))

    classes = config['classes']
    conf = config['conf']
    tracker = config['tracker']
    ver = config['ver']
    ckpt = config['model']

    lines = config['lines']
    lines = [
        LineZone(
            start=Point(i[0][0], i[0][1]),
            end=Point(i[1][0], i[1][1]),
        )
        for i in lines
    ]
    polygons = config['polygons']
    polygons = [np.array(i) for i in polygons]
    (
        line_annotator,
        zones,
        zone_annotators,
        box,
        skip_label,
        mask,
        mask_opacity,
        area,
        predict_color,
        show_fps,
    ) = init_annotator(config, reso, polygons)

    legacy = ver == 'v5'
    if ver == 'rtdetr':
        m = RTDETR(ckpt)
    else:
        m = YOLO(ckpt) if not legacy else yolov5.load(ckpt)

    allclasses = m.model.names

    if legacy:
        m.classes = classes
        m.conf = conf

    def model(source, classes=classes, conf=conf, tracker=None):
        return (
            m.predict(source, classes=classes, conf=conf, retina_masks=True)
            if tracker is None
            else m.track(
                source,
                classes=classes,
                conf=conf,
                retina_masks=True,
                tracker=f'{tracker}.yaml',
            )
        )

    def infer(f):
        return annot(
            f,
            m if legacy else model,
            tracker,
            legacy,
            allclasses,
            lines,
            line_annotator,
            zones,
            zone_annotators,
            box,
            skip_label,
            mask,
            mask_opacity,
            area,
            predict_color,
            show_fps,
        )[0]

    options = {
        'CAP_PROP_FOURCC': cv2.VideoWriter.fourcc(*'MJPG'),
        'CAP_PROP_FPS': fps,
        'CAP_PROP_FRAME_WIDTH': reso[0],
        'CAP_PROP_FRAME_HEIGHT': reso[1],
    }
    stream = VideoGear(source=source, **options).start()

    if saveto is None:
        while True:
            cv2.imshow('', infer(stream.read()))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        writer = WriteGear(output=saveto)
        while True:
            writer.write(infer(stream.read()))

    cv2.destroyAllWindows()
    stream.stop()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--saveto', type=str, default=None)
    args = parser.parse_args()
    source = args.source
    config = args.config
    saveto = args.saveto
    app(source, config, saveto)
