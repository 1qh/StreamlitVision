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
    VideoSink,
    draw_text,
    get_video_frames_generator,
)
from ultralytics import YOLO

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


ycc_colors = rgb2ycc(colors_rgb)


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
    predict_color,
):
    det = Detections.from_yolov8(res)
    if res.boxes.id is not None:
        det.tracker_id = res.boxes.id.cpu().numpy().astype(int)
    if res.masks is not None:
        det.mask = res.masks.data.cpu().numpy()
    f = res.orig_img

    if predict_color:
        n = det.xyxy.shape[0]
        centers_color = np.zeros((n, 3), dtype=np.uint8)

        centers_x = np.mean(det.xyxy[:, [0, 2]], axis=1).astype(int)
        centers_y = np.mean(det.xyxy[:, [1, 3]], axis=1).astype(int)

        centers_color = f[centers_y, centers_x]

        for i in range(n):
            rgb = centers_color[i]
            predict = closest(rgb, ycc_colors)
            color = colors_rgb[predict][::-1]
            color = tuple(map(int, color))
            color_name = colors[predict]
            cv2.putText(
                f,
                color_name,
                (centers_x[i], centers_y[i]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

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


def app(source, config='config.json', saveto=None):
    cams = [i for i in range(-1, 2, 1)]
    if '.' not in source and int(source) in cams:
        source = int(source)
        width, height = maxcam()
    else:
        vid = VideoInfo.from_video_path(source)
        width, height = vid.resolution_wh

    config = json.load(open(config))
    classes = config['classes']
    conf = config['conf']
    tracker = config['tracker']
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
    predict_color = config['predict_color'] if 'predict_color' in config else False
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

    def callback(f):
        return annot(
            allclasses,
            model(f, tracker=tracker)[0],
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

    if type(source) == int:
        cap = cv2.VideoCapture(0)
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(6, codec)
        cap.set(5, 30)
        cap.set(3, width)
        cap.set(4, height)
        if saveto is None:
            while True:
                cv2.imshow('', callback(cap.read()[1]))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            writer = cv2.VideoWriter(saveto, codec, 30, (width, height))
            while True:
                writer.write(callback(cap.read()[1]))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            writer.release()

        cap.release()
        cv2.destroyAllWindows()
    else:
        if saveto is None:
            for res in model(
                source,
                stream=True,
                tracker=tracker,
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
                    predict_color,
                )
                cv2.imshow('', f)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            save(source, saveto, callback)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--saveto', type=str, default=None)
    args = parser.parse_args()
    source = args.source
    config = args.config
    saveto = args.saveto
    app(source, config, saveto)
