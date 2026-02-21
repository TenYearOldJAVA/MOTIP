# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import glob
import torch
from collections import defaultdict
from configparser import ConfigParser

from .one_dataset import OneDataset
from .util import is_legal, append_annotation


class DanceTrack(OneDataset):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "DanceTrack",
            split: str = "train",
            load_annotation: bool = True,
    ):
        super(DanceTrack, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )

        # Prepare the data:
        self.sequence_infos = self._get_sequence_infos()
        self.image_paths = self._get_image_paths()
        if self.load_annotation:
            self.annotations = self._get_annotations()
        return

    def _get_sequence_names(self):
        split_dir = os.path.join(self.data_dir, self.split)
        names = [
            n for n in sorted(os.listdir(split_dir))
            if os.path.isdir(os.path.join(split_dir, n))
            and os.path.isdir(os.path.join(split_dir, n, "img1"))
        ]
        # 训练/验证需要标注：只保留有 gt/gt.txt 的序列（test 集官方不公开 gt，会没有）
        if self.load_annotation:
            names = [
                n for n in names
                if os.path.isfile(os.path.join(split_dir, n, "gt", "gt.txt"))
            ]
        return names

    def _get_sequence_infos(self):
        sequence_names = self._get_sequence_names()
        sequence_infos = dict()
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            ini_path = os.path.join(sequence_dir, "seqinfo.ini")
            width, height, length = None, None, None
            if os.path.isfile(ini_path):
                ini = ConfigParser()
                ini.read(ini_path, encoding="utf-8")
                for sec in ("Sequence", "sequence"):
                    if ini.has_section(sec):
                        try:
                            width = int(ini.get(sec, "imWidth", fallback=ini.get(sec, "width", fallback=None)))
                            height = int(ini.get(sec, "imHeight", fallback=ini.get(sec, "height", fallback=None)))
                            length = int(ini.get(sec, "seqLength", fallback=ini.get(sec, "length", fallback=None)))
                        except (TypeError, ValueError):
                            pass
                        break
            if length is None:
                img_dir = os.path.join(sequence_dir, "img1")
                if os.path.isdir(img_dir):
                    length = len(glob.glob(os.path.join(img_dir, "*.jpg")))
                if length is None:
                    length = 0
            if width is None or height is None:
                first_img = os.path.join(sequence_dir, "img1", "00000001.jpg")
                if os.path.isfile(first_img):
                    try:
                        from PIL import Image
                        with Image.open(first_img) as im:
                            width, height = im.size
                    except Exception:
                        width, height = 1920, 1080
                else:
                    width, height = 1920, 1080
            sequence_infos[sequence_name] = {
                "width": width,
                "height": height,
                "length": length,
                "is_static": False,
            }
        return sequence_infos

    def _get_image_paths(self):
        sequence_names = self._get_sequence_names()
        image_paths = defaultdict(list)
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            for i in range(self.sequence_infos[sequence_name]["length"]):
                image_paths[sequence_name].append(self._get_image_path(sequence_dir, i))
        return image_paths

    @staticmethod
    def _get_sequence_dir(data_dir, split, sequence_name):
        return str(os.path.join(data_dir, split, sequence_name))

    @staticmethod
    def _get_image_path(sequence_dir, frame_idx):
        return str(os.path.join(sequence_dir, "img1", f"{frame_idx+1:08d}.jpg"))    # the image name is 1-indexed

    def _get_annotations(self):
        sequence_names = self._get_sequence_names()
        # Init the annotations:
        annotations = self._init_annotations(sequence_names)
        # Load the annotations:
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            gt_file_path = os.path.join(sequence_dir, "gt", "gt.txt")
            with open(gt_file_path, "r") as gt_file:
                for line in gt_file:
                    line = line.strip().split(",")
                    frame_id, obj_id, x, y, w, h, _, _, _ = line
                    frame_id, obj_id = map(int, [frame_id, obj_id])
                    x, y, w, h = map(float, [x, y, w, h])
                    bbox = [x, y, w, h]
                    category, visibility = 0, 1.0
                    ann_index = frame_id - 1    # 0-indexed for annotations
                    # Organized into the annotations:
                    annotations[sequence_name][ann_index] = append_annotation(
                        annotation=annotations[sequence_name][ann_index],
                        obj_id=obj_id,
                        category=category,
                        bbox=bbox,
                        visibility=visibility,
                    )
        # Determine whether each annotation is legal:
        for sequence_name in sequence_names:
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name][i]["is_legal"] = is_legal(annotations[sequence_name][i])
        return annotations

    def _init_annotations(self, sequence_names):
        annotations = dict()
        for sequence_name in sequence_names:
            annotations[sequence_name] = []
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name].append({
                    "id": torch.zeros((0, ), dtype=torch.int64),
                    "category": torch.zeros((0, ), dtype=torch.int64),
                    "bbox": torch.zeros((0, 4), dtype=torch.float32),
                    "visibility": torch.zeros((0, ), dtype=torch.float32),
                })
        return annotations
