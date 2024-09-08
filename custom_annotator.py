import numpy as np
from supervision import (
  BackgroundOverlayAnnotator,
  BlurAnnotator,
  BoundingBoxAnnotator,
  BoxAnnotator,
  BoxCornerAnnotator,
  CircleAnnotator,
  Color,
  ColorAnnotator,
  ColorPalette,
  CropAnnotator,
  Detections,
  DotAnnotator,
  EdgeAnnotator,
  EllipseAnnotator,
  FPSMonitor,
  HaloAnnotator,
  HeatMapAnnotator,
  LabelAnnotator,
  LineZone,
  LineZoneAnnotator,
  MaskAnnotator,
  OrientedBoxAnnotator,
  PercentageBarAnnotator,
  PixelateAnnotator,
  Point,
  PolygonAnnotator,
  PolygonZone,
  PolygonZoneAnnotator,
  RichLabelAnnotator,
  RoundBoxAnnotator,
  TraceAnnotator,
  TriangleAnnotator,
  VertexAnnotator,
  VertexLabelAnnotator,
  crop_image,
  draw_text,
  get_polygon_center,
)

from utils import ColorClassifier, Draw, avg_rgb, i8, plur


class ColorClassifierAnnotator:
  def __init__(
    self,
    clf: ColorClassifier = ColorClassifier(),
    naive: bool = False,
    text_color: Color = Color.BLACK,
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 10,
  ):
    self.naive: bool = naive
    self.text_color: Color = text_color
    self.text_scale: float = text_scale
    self.text_thickness: int = text_thickness
    self.text_padding: int = text_padding
    if clf.names:
      self.rgb_colors = clf.rgb
      self.color_names = clf.names or []
      self.clf = clf

  def annotate(
    self,
    scene: i8,
    detections: Detections,
  ) -> i8:
    xyxy = detections.xyxy.astype(int)
    centers = (xyxy[:, [0, 1]] + xyxy[:, [2, 3]]) // 2

    for center, bb in zip(centers, xyxy):
      x = center[0]
      y = center[1]

      # for shirt color of person
      # w = bb[2] - bb[0]
      # h = bb[3] - bb[1]
      # cropped = f[
      #     bb[1] : bb[3] - int(h * 0.4),
      #     bb[0] + int(w * 0.2) : bb[2] - int(w * 0.2),
      # ]

      cropped: i8 = crop_image(scene, bb)
      if 0 in cropped.shape:
        continue

      rgb = scene[y, x] if self.naive else avg_rgb(cropped)
      predict = self.clf.closest(rgb)
      r, g, b = self.rgb_colors[predict]
      draw_text(
        scene=scene,
        text=self.color_names[predict],
        text_anchor=Point(x=x, y=y + 20),
        text_color=Color(255 - r, 255 - g, 255 - b),
        text_scale=self.text_scale,
        text_padding=self.text_padding,
        background_color=Color(r, g, b),
      )
    return scene


class FpsAnnotator:
  def __init__(
    self,
    text_anchor: Point = Point(x=50, y=20),
    text_color: Color = Color.BLACK,
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 1,
    sample_size: int = 10,
  ):
    self.text_anchor: Point = text_anchor
    self.text_color: Color = text_color
    self.text_scale: float = text_scale
    self.text_thickness: int = text_thickness
    self.text_padding: int = text_padding
    self.fps_monitor = FPSMonitor(sample_size)

  def annotate(self, scene: i8, _) -> i8:
    self.fps_monitor.tick()
    draw_text(
      scene=scene,
      text=f'{self.fps_monitor.fps:.2f}',
      text_anchor=self.text_anchor,
      text_color=self.text_color,
      text_scale=self.text_scale * 2,
      text_thickness=self.text_thickness,
      text_padding=self.text_padding,
    )
    return scene


class CountAnnotator:
  def __init__(
    self,
    names: list[str] | None = None,
    text_anchor: Point = Point(x=50, y=12),
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 1,
  ):
    if names is None:
      names = []
    self.names = names
    self.text_anchor: Point = text_anchor
    self.text_scale: float = text_scale
    self.text_thickness: int = text_thickness
    self.text_padding: int = text_padding
    self.pallet: ColorPalette = ColorPalette.DEFAULT

  def annotate(self, scene: i8, detections: Detections) -> i8:
    names = self.names
    class_id: i8 | None = detections.class_id
    if len(names) and class_id is not None:
      for i, c in enumerate(np.bincount(class_id)):
        if c:
          bg = self.pallet.by_idx(i)
          draw_text(
            scene=scene,
            text=plur(c, names[i]),
            text_anchor=Point(
              x=scene.shape[1] - self.text_anchor.x,
              y=self.text_anchor.y + int(i * self.text_scale * 18),
            ),
            text_color=Color.WHITE if np.sum(bg.as_bgr()) < 384 else Color.BLACK,
            text_scale=self.text_scale,
            text_thickness=self.text_thickness,
            text_padding=self.text_padding,
            background_color=bg,
          )
    return scene


class AreaAnnotator:
  def __init__(
    self,
    text_color: Color = Color.BLACK,
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 1,
  ):
    self.text_color: Color = text_color
    self.text_scale: float = text_scale
    self.text_thickness: int = text_thickness
    self.text_padding: int = text_padding

  def annotate(self, scene: i8, detections: Detections) -> i8:
    xyxy = detections.xyxy.astype(int)
    centers = (xyxy[:, [0, 1]] + xyxy[:, [2, 3]]) // 2
    for a, c in zip(detections.area.astype(int), centers):
      draw_text(
        scene=scene,
        text=str(a)[:-1],
        text_anchor=Point(x=c[0], y=c[1]),
        text_color=self.text_color,
        text_scale=self.text_scale,
        text_thickness=self.text_thickness,
        text_padding=self.text_padding,
      )
    return scene


class LineAndZoneAnnotator:
  def __init__(
    self,
    draw: Draw = Draw(),
    wh: tuple[int, int] = (640, 480),
    thickness: int = 2,
    text_thickness: int = 2,
    text_color: Color = Color.BLACK,
    text_scale: float = 0.5,
    text_offset: float = 1.5,
    text_padding: int = 1,
  ):
    self.draw: Draw = draw
    self.wh: tuple[int, int] = wh
    self.ls: list[LineZone] = [
      LineZone(
        start=Point(i[0][0], i[0][1]),
        end=Point(i[1][0], i[1][1]),
      )
      for i in self.draw.lines
    ]
    self.line = LineZoneAnnotator(
      thickness=thickness,
      text_thickness=text_thickness,
      text_color=text_color,
      text_scale=text_scale,
      text_offset=text_offset,
      text_padding=text_padding,
    )
    self.zs: list[PolygonZone] = [
      PolygonZone(polygon=np.array(p), frame_resolution_wh=self.wh) for p in self.draw.zones
    ]
    self.origin_zs = self.zs
    self.zones: list[PolygonZoneAnnotator] = [
      PolygonZoneAnnotator(
        zone=z,
        color=ColorPalette.DEFAULT.by_idx(i),
        thickness=thickness,
        text_color=text_color,
        text_scale=text_scale,
        text_thickness=text_thickness,
        text_padding=text_padding,
      )
      for i, z in enumerate(self.zs)
    ]

  def annotate(self, scene: i8, detections: Detections) -> i8:
    for l in self.ls:  # noqa: E741
      l.trigger(detections)
      self.line.annotate(frame=scene, line_counter=l)

    for z, zone in zip(self.zs, self.zones):
      z.trigger(detections)
      zone.annotate(scene)

    return scene

  def update(self, f: i8):
    scale = f.shape[0] / self.wh[1]
    self.ls = [
      LineZone(
        start=Point(i[0][0] * scale, i[0][1] * scale),
        end=Point(i[1][0] * scale, i[1][1] * scale),
      )
      for i in self.draw.lines
    ]
    self.zs = [
      PolygonZone(
        polygon=(z.polygon * scale).astype(int),
        frame_resolution_wh=(f.shape[1], f.shape[0]),
      )
      for z in self.origin_zs
    ]
    for i, z in enumerate(self.zs):
      self.zones[i].zone = z
      self.zones[i].center = get_polygon_center(polygon=z.polygon)


all_anns = {
  AreaAnnotator,
  BackgroundOverlayAnnotator,
  BlurAnnotator,
  BoundingBoxAnnotator,
  BoxAnnotator,
  BoxCornerAnnotator,
  CircleAnnotator,
  ColorAnnotator,
  ColorClassifierAnnotator,
  CountAnnotator,
  CropAnnotator,
  DotAnnotator,
  EdgeAnnotator,
  EllipseAnnotator,
  FpsAnnotator,
  HaloAnnotator,
  HeatMapAnnotator,
  LabelAnnotator,
  LineAndZoneAnnotator,
  MaskAnnotator,
  OrientedBoxAnnotator,
  PercentageBarAnnotator,
  PixelateAnnotator,
  PolygonAnnotator,
  RichLabelAnnotator,
  RoundBoxAnnotator,
  TraceAnnotator,
  TriangleAnnotator,
  VertexAnnotator,
  VertexLabelAnnotator,
}
all_class = {i.__name__.replace('Annotator', ''): i for i in all_anns}
