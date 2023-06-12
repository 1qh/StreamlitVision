#!/usr/bin/env python3
import cv2
from supervision import VideoInfo
from typer import run
from vidgear.gears import WriteGear

from core import Annotator, maxcam


def app(source=0, config='config.json', saveto=None):
    if '.' not in source and int(source) in [i for i in range(-1, 2, 1)]:
        source = int(source)
        reso = maxcam()
    else:
        vid = VideoInfo.from_video_path(source)
        reso = vid.resolution_wh

    an = Annotator.load(config, reso)

    writer = WriteGear(output=saveto if saveto is not None else 'out.mp4')
    if saveto is None:
        for f, _ in an.gen(source):
            cv2.imshow('', f)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        for f, _ in an.gen(source):
            writer.write(f)

    cv2.destroyAllWindows()
    writer.close()


if __name__ == '__main__':
    run(app)
