#!/usr/bin/env python3
import argparse
import json
from subprocess import check_output

import cv2
import numpy as np
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
)
from ultralytics import YOLO


def cvt(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def annot(
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
):
    det = Detections.from_yolov8(res)
    if res.boxes.id is not None:
        det.tracker_id = res.boxes.id.cpu().numpy().astype(int)
    if res.masks is not None:
        det.mask = res.masks.data.cpu().numpy()
    f = res.orig_img

    if box:
        f = box.annotate(
            scene=f,
            detections=det,
            labels=[
                f'{conf:0.2f} {allclasses[cls]}' + (f' {track_id}' if track_id else '')
                for _, _, conf, cls, track_id in det
            ],
        )
    if mask:
        f = mask.annotate(
            scene=f,
            detections=det,
            opacity=mask_opacity,
        )
    if mask and area:
        for t, a in zip(det.area, det.xyxy.astype(int)):
            draw_text(
                scene=f,
                text=f'{int(t)}',
                text_anchor=Point(x=(a[0] + a[2]) // 2, y=(a[1] + a[3]) // 2),
                text_color=line_annotator.text_color,
                text_scale=line_annotator.text_scale,
                text_padding=line_annotator.text_padding,
            )
    for l in lines:
        l.trigger(detections=det)
        line_annotator.annotate(frame=f, line_counter=l)

    for z, zone in zip(zones, zone_annotators):
        z.trigger(detections=det)
        f = zone.annotate(scene=f)
    return f


def app(source, config='config.json'):
    cams = [i for i in range(-1, 2, 1)]
    if '.' not in source and int(source) in cams:
        source = int(source)
        reso = check_output(
            "v4l2-ctl -d /dev/video0 --list-formats-ext | grep Size: | tail -1 | awk '{print $NF}'",
            shell=True,
        )
        width, height = [int(i) for i in reso.decode().split('x')]
    else:
        vid = VideoInfo.from_video_path(source)
        width, height = vid.resolution_wh

    config = json.load(open(config))
    classes = config['classes']
    conf = config['conf']
    track = config['track']
    ckpt = config['model']
    visual = config['visual']
    lines = config['lines']
    polygons = config['polygons']
    thickness = visual['thickness']
    text_scale = visual['text_scale']
    text_offset = visual['text_offset']
    text_padding = visual['text_padding']
    text_color = visual['text_color']
    use_box = config['box'] if 'box' in config else False
    use_mask = config['mask'] if 'mask' in config else False
    area = config['area'] if 'area' in config else False
    lines = [
        LineZone(
            start=Point(i[0][0], i[0][1]),
            end=Point(i[1][0], i[1][1]),
        )
        for i in lines
    ]
    polygons = [np.array(i) for i in polygons]

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

    m = YOLO(ckpt)
    allclasses = m.model.names

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

    if type(source) == int:
        cap = cv2.VideoCapture(0)
        codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        cap.set(6, codec)
        cap.set(5, 30)
        cap.set(3, width)
        cap.set(4, height)
        while True:
            _, f = cap.read()
            res = model(f, track=track)[0]
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
            )
            cv2.imshow('', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        for res in model(
            source,
            stream=True,
            track=track,
        ):
            f = annot(
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
            )
            cv2.imshow('', f)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()
    source = args.source
    config = args.config
    app(source, config)
