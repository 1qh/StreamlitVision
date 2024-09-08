import json
from collections.abc import Iterator

import numpy as np
from PIL import Image
from streamlit import image
from streamlit.runtime.uploaded_file_manager import UploadedFile
from supervision import Detections, KeyPoints
from vidgear.gears import VideoGear

from custom_annotator import LineAndZoneAnnotator, all_class
from utils import from_plain, i8, weight2model


class Model:
  def __init__(self, weight: str, options: dict):
    model = weight2model(weight)
    if hasattr(model, 'names'):
      self.names: list[str] = model.names
    self.task: str | None = model.task
    self.options = dict(retina_masks=True) | options
    self.run = model.track if self.options.get('tracker') else model.predict
    self.predict = model.predict

  def gen(self, source: str | int) -> Iterator[tuple[i8, tuple[Detections | KeyPoints | None, i8]]]:
    stream = VideoGear(source=source).start()
    while (f := stream.read()) is not None:
      yield f, self(f)

  def __call__(self, f: i8) -> tuple[Detections | KeyPoints | None, i8]:
    res = self.run(f, **self.options)[0]
    return (
      (Detections.from_ultralytics(res) if res.boxes is not None else None)
      if self.task != 'pose'
      else KeyPoints.from_ultralytics(res)
    ), res.plot(line_width=2, kpt_radius=1)

  def predict_image(self, file: UploadedFile):
    image(
      self.predict(
        np.array(Image.open(file)),
        **{k: v for k, v in self.options.items() if k != 'persist'},
      )[0].plot(line_width=3, kpt_radius=1),
      use_column_width='always',
    )


class Annotator:
  def __init__(self, model: Model, config: dict | None = None, wh: tuple[int, int] = (640, 480)):
    if config is None:
      config = {}

    config = from_plain(config)
    anns = {i: all_class[i](**config[i]) for i in config}

    self.model = model
    if hasattr(model, 'names'):
      self.names: list[str] = model.names

    if 'Label' in anns:
      self.label = anns['Label']
      del anns['Label']
    else:
      self.label = None

    if 'Trace' in anns:
      self.trace = anns['Trace']
      del anns['Trace']
    else:
      self.trace = None

    self.linezone: LineAndZoneAnnotator | None = None
    if 'LineAndZone' in anns:
      self.linezone = anns['LineAndZone']

    self.anns = anns
    self.wh = wh

  @classmethod
  def load(cls, path: str):
    with open(path, encoding='utf-8') as f:
      d = json.load(f)
    return cls(model=Model(d['weight'], d['options']), config=d['config'], wh=d['wh'])

  def one(self, f: i8, det: Detections | KeyPoints | None) -> i8:
    names = self.names
    if det and self.label:
      f = self.label.annotate(
        f,
        det,
        labels=[
          f'{conf:0.2f} {names[cl or 0] if len(names) else cl}' + (f' {track_id}' if track_id else '')
          for _, _, conf, cl, track_id, _ in det
        ],
      )
    if self.trace:
      try:
        f = self.trace.annotate(f, det)
      except Exception as e:
        print(e)

    for v in self.anns.values():
      f = v.annotate(f, det)
    return f

  def gen(self, source: str | int) -> Iterator[i8]:
    for f, (det, _) in self.model.gen(source):
      yield self.one(f, det)

  def __call__(self, f: i8) -> tuple[i8, i8]:
    det, fallback = self.model(f)
    return self.one(f, det), fallback
