# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for AVA evaluation."""

from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import logging
import pprint
import time
from collections import defaultdict

import numpy as np

import slowfast.utils.distributed as du
from slowfast.utils.env import pathmgr
from ava_evaluation import (
    object_detection_evaluation,
    standard_fields,
)

logger = logging.getLogger(__name__)


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return "%s,%04d" % (video_id, int(timestamp))


def read_csv(csv_file, class_whitelist=None, load_score=False):
    """Loads boxes and class labels from a CSV file in the AVA format."""
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    with pathmgr.open(csv_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            assert len(row) in [7, 8], "Wrong number of columns: " + str(row)
            image_key = make_image_key(row[0], row[1])
            x1, y1, x2, y2 = [float(n) for n in row[2:6]]
            action_id = int(row[6])
            if class_whitelist and action_id not in class_whitelist:
                continue
            score = 1.0
            if load_score:
                score = float(row[7])
            boxes[image_key].append([y1, x1, y2, x2])
            labels[image_key].append(action_id)
            scores[image_key].append(score)
    return boxes, labels, scores


def read_exclusions(exclusions_file):
    """Reads a CSV file of excluded timestamps."""
    excluded = set()
    if exclusions_file:
        with pathmgr.open(exclusions_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                assert len(row) == 2, "Expected only 2 columns, got: " + str(row)
                excluded.add(make_image_key(row[0], row[1]))
    return excluded


def read_labelmap(labelmap_file):
    """Read label map and class ids."""
    labelmap = []
    class_ids = set()
    name = ""
    class_id = ""
    with pathmgr.open(labelmap_file, "r") as f:
        for line in f:
            if line.startswith("  name:"):
                name = line.split('"')[1]
            elif line.startswith("  id:") or line.startswith("  label_id:"):
                class_id = int(line.strip().split(" ")[-1])
                labelmap.append({"id": class_id, "name": name})
                class_ids.add(class_id)
    return labelmap, class_ids


def evaluate_ava_from_files(labelmap, groundtruth, detections, exclusions):
    """Run AVA evaluation given annotation/prediction files."""
    categories, class_whitelist = read_labelmap(labelmap)
    excluded_keys = read_exclusions(exclusions)
    groundtruth = read_csv(groundtruth, class_whitelist, load_score=False)
    detections = read_csv(detections, class_whitelist, load_score=True)
    run_evaluation(categories, groundtruth, detections, excluded_keys)


def evaluate_ava(
    preds,
    original_boxes,
    metadata,
    excluded_keys,
    class_whitelist,
    categories,
    groundtruth=None,
    video_idx_to_name=None,
    name="latest",
):
    """Run AVA evaluation given numpy arrays."""
    eval_start = time.time()

    detections = get_ava_eval_data(
        preds,
        original_boxes,
        metadata,
        class_whitelist,
        video_idx_to_name=video_idx_to_name,
    )

    logger.info("Evaluating with %d unique GT frames." % len(groundtruth[0]))
    logger.info("Evaluating with %d unique detection frames" % len(detections[0]))

    write_results(detections, "detections_%s.csv" % name)
    write_results(groundtruth, "groundtruth_%s.csv" % name)

    results = run_evaluation(categories, groundtruth, detections, excluded_keys)

    logger.info("AVA eval done in %f seconds." % (time.time() - eval_start))
    return results["PascalBoxes_Precision/mAP@0.5IOU"]


def run_evaluation(categories, groundtruth, detections, excluded_keys, verbose=True):
    """AVA evaluation main logic."""
    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(categories)

    boxes, labels, _ = groundtruth
    for image_key in boxes:
        if image_key in excluded_keys:
            logger.info(
                "Found excluded timestamp in ground truth: %s. It will be ignored.",
                image_key,
            )
            continue
        pascal_evaluator.add_single_ground_truth_image_info(
            image_key,
            {
                standard_fields.InputDataFields.groundtruth_boxes: np.array(
                    boxes[image_key], dtype=float
                ),
                standard_fields.InputDataFields.groundtruth_classes: np.array(
                    labels[image_key], dtype=int
                ),
                standard_fields.InputDataFields.groundtruth_difficult: np.zeros(
                    len(boxes[image_key]), dtype=bool
                ),
            },
        )

    boxes, labels, scores = detections
    for image_key in boxes:
        if image_key in excluded_keys:
            logger.info(
                "Found excluded timestamp in detections: %s. It will be ignored.",
                image_key,
            )
            continue
        pascal_evaluator.add_single_detected_image_info(
            image_key,
            {
                standard_fields.DetectionResultFields.detection_boxes: np.array(
                    boxes[image_key], dtype=float
                ),
                standard_fields.DetectionResultFields.detection_classes: np.array(
                    labels[image_key], dtype=int
                ),
                standard_fields.DetectionResultFields.detection_scores: np.array(
                    scores[image_key], dtype=float
                ),
            },
        )

    metrics = pascal_evaluator.evaluate()

    if du.is_master_proc():
        pprint.pprint(metrics, indent=2)
    return metrics


def get_ava_eval_data(
    scores,
    boxes,
    metadata,
    class_whitelist,
    verbose=False,
    video_idx_to_name=None,
):
    """Convert our data format into the data format used in official AVA evaluation."""
    out_scores = defaultdict(list)
    out_labels = defaultdict(list)
    out_boxes = defaultdict(list)
    for i in range(scores.shape[0]):
        video_idx = int(np.round(metadata[i][0]))
        sec = int(np.round(metadata[i][1]))

        video = video_idx_to_name[video_idx]

        key = f"{video},{sec:04d}"
        batch_box = boxes[i].tolist()
        batch_box = [batch_box[j] for j in [0, 2, 1, 4, 3]]

        one_scores = scores[i].tolist()
        for cls_idx, score in enumerate(one_scores):
            if cls_idx + 1 in class_whitelist:
                out_scores[key].append(score)
                out_labels[key].append(cls_idx + 1)
                out_boxes[key].append(batch_box[1:])

    return out_boxes, out_labels, out_scores


def write_results(detections, filename):
    """Write prediction results into official formats."""
    start = time.time()

    boxes, labels, scores = detections
    with pathmgr.open(filename, "w") as f:
        for key in boxes.keys():
            for box, label, score in zip(boxes[key], labels[key], scores[key]):
                f.write(
                    "%s,%.03f,%.03f,%.03f,%.03f,%d,%.04f\n"
                    % (key, box[1], box[0], box[3], box[2], label, score)
                )

    logger.info("AVA results wrote to %s" % filename)
    logger.info("\ttook %d seconds." % (time.time() - start))
