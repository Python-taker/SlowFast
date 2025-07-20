#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import os
import pickle

import numpy as np
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, time, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            if isinstance(inputs, list):
                inputs = [inp.cuda(non_blocking=True) for inp in inputs]
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, list):
                    meta[key] = [v.cuda(non_blocking=True) for v in val]
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds, ori_boxes, metadata = [
                    x.detach().cpu() for x in (preds, ori_boxes, metadata)
                ]

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        else:
            preds = model(inputs)
            if cfg.NUM_GPUS > 1:
                preds, labels, video_idx = du.all_gather([preds, labels, video_idx])
            if cfg.NUM_GPUS:
                preds, labels, video_idx = [x.cpu() for x in (preds, labels, video_idx)]

            test_meter.iter_toc()
            if not cfg.VIS_MASK.ENABLE:
                test_meter.update_stats(preds.detach(), labels.detach(), video_idx.detach())
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds, all_labels = all_preds.cpu(), all_labels.cpu()
        if writer:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH:
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)
            if du.is_root_proc():
                with pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)
            logger.info(f"Saved prediction results to {save_path}")

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    """
    # ✅ 수정된 부분: NUM_GPUS가 1보다 클 때만 distributed 초기화
    if cfg.NUM_GPUS > 1:
        du.init_distributed_training(cfg)

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    logging.setup_logging(cfg.OUTPUT_DIR)

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    test_meters = []
    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:
        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view

        logger.info("Test with config:")
        logger.info(cfg)

        model = build_model(cfg)
        flops, params = 0.0, 0.0
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            model.eval()
            flops, params = misc.log_model_info(model, cfg, use_train_input=False)

        cu.load_test_checkpoint(cfg, model)

        test_loader = loader.construct_loader(cfg, "test")
        logger.info(f"Testing model for {len(test_loader)} iterations")

        if cfg.DETECTION.ENABLE:
            assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
            test_meter = AVAMeter(len(test_loader), cfg, mode="test")
        else:
            assert test_loader.dataset.num_videos % (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            ) == 0
            test_meter = TestMeter(
                test_loader.dataset.num_videos // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES,
                len(test_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )

        writer = tb.TensorboardWriter(cfg) if cfg.TENSORBOARD.ENABLE and du.is_master_proc() else None

        test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
        test_meters.append(test_meter)
        if writer:
            writer.close()

    result_string_views = f"_p{params/1e6:.2f}_f{flops:.2f}"
    for view, meter in zip(cfg.TEST.NUM_TEMPORAL_CLIPS, test_meters):
        logger.info(
            f"Finalized testing with {view} temporal clips and {cfg.TEST.NUM_SPATIAL_CROPS} spatial crops"
        )
        result_string_views += f"_{view}a{meter.stats['top1_acc']}"

        result_string = (
            f"_p{params/1e6:.2f}_f{flops:.2f}_{view}a{meter.stats['top1_acc']} "
            f"Top5 Acc: {meter.stats['top5_acc']} MEM: {misc.gpu_mem_usage():.2f} f: {flops:.4f}"
        )

        logger.info(result_string)

    logger.info(result_string_views)
    return result_string + " \n " + result_string_views
