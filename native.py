#!/usr/bin/env python3
import cv2
from supervision import VideoInfo
from typer import run
from vidgear.gears import VideoGear, WriteGear

from core import Annotator, maxcam


def app(source=0, config='config.json', saveto=None):
    if '.' not in source and int(source) in [i for i in range(-1, 2, 1)]:
        source = int(source)
        reso = maxcam()
        fps = 30
    else:
        vid = VideoInfo.from_video_path(source)
        reso = vid.resolution_wh
        fps = vid.fps

    an = Annotator.load(config, reso)
    options = {
        'CAP_PROP_FOURCC': cv2.VideoWriter.fourcc(*'MJPG'),
        'CAP_PROP_FPS': fps,
        'CAP_PROP_FRAME_WIDTH': reso[0],
        'CAP_PROP_FRAME_HEIGHT': reso[1],
    }
    stream = VideoGear(source=source, **options).start()
    if saveto is None:
        while True:
            cv2.imshow('', an(stream.read())[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        writer = WriteGear(output=saveto)
        while True:
            writer.write(an(stream.read())[0])

    cv2.destroyAllWindows()
    stream.stop()
    writer.close()


if __name__ == '__main__':
    run(app)
