import pprint
from tqdm import tqdm, trange
import numpy as np
import os
from collections import defaultdict
from utils.basic_utils import AverageMeter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from FlashVTG.config import TestOptions
from FlashVTG.start_end_dataset import (
    StartEndDataset,
    start_end_collate,
    prepare_batch_inputs,
)
from FlashVTG.postprocessing import PostProcessorDETR
from standalone_eval.eval import eval_submission
from utils.basic_utils import save_jsonl, save_json

import nncore
from nncore.ops import temporal_iou

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


# for MR
@torch.no_grad()
def compute_mr_results(model, eval_loader, opt):
    model.eval()

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]

        model_inputs, targets = prepare_batch_inputs(
            batch[1], opt.device, non_blocking=opt.pin_memory
        )

        if targets is not None:
            targets["label"] = batch[0]
            targets["fps"] = torch.full((256,), 1 / opt.clip_length).to(
                opt.device
            )  # if datasets is qv, fps is 0.5, charades' is 1
        else:
            targets = {}
        outputs = model(**model_inputs, targets=targets)

        if opt.span_loss_type == "l1":
            scores = outputs["_out"]["boundary"][:, 2]
            pred_spans = outputs["_out"]["boundary"][:, :2].unsqueeze(0)
            _saliency_scores = outputs["_out"]["saliency"].unsqueeze(0)

            saliency_scores = []
            valid_vid_lengths = outputs["_out"]["video_msk"].sum(1).cpu().tolist()
            for j in range(len(valid_vid_lengths)):
                ss = _saliency_scores[j, : int(valid_vid_lengths[j])].tolist()
                ss = [float(f"{e:.4f}") for e in ss]
                saliency_scores.append(ss)
        else:
            bsz, n_queries = outputs["pred_spans"].shape[
                :2
            ]  # # (bsz, #queries, max_v_l *2)
            pred_spans_logits = outputs["pred_spans"].view(
                bsz, n_queries, 2, opt.max_v_l
            )
            pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(
                -1
            )  # 2 * (bsz, #queries, 2)
            scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
            pred_spans[:, 1] += 1
            pred_spans *= opt.clip_length

        # compose predictions
        for idx, (meta, spans, score) in enumerate(
            zip(query_meta, pred_spans.cpu(), scores.cpu())
        ):
            spans = torch.clamp(outputs["_out"]["boundary"], 0, meta["duration"])
            cur_ranked_preds = spans.tolist()
            cur_ranked_preds = [
                [float(f"{e:.4f}") for e in row] for row in cur_ranked_preds
            ]
            cur_query_pred = dict(
                qid=meta["qid"],
                query=meta["query"],
                vid=meta["vid"],
                pred_relevant_windows=cur_ranked_preds,
                pred_saliency_scores=saliency_scores[idx],
            )
            mr_res.append(cur_query_pred)

    post_processor = PostProcessorDETR(
        clip_length=opt.clip_length,
        min_ts_val=0,
        max_ts_val=150,
        min_w_l=2,
        max_w_l=150,
        move_window_method="left",
        process_func_names=("clip_ts", "round_multiple"),
    )

    mr_res = post_processor(mr_res)
    return mr_res


def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    from FlashVTG.model import build_model1

    model, _ = build_model1(opt)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    if opt.resume_adapter is not None:
        logger.info(f"Load adapter checkpoint from {opt.resume_adapter}")
        adapter_checkpoint = torch.load(opt.resume_adapter)
        adapter_state_dict = {
            k: v
            for k, v in adapter_checkpoint["state_dict"].items()
            if k.startswith("adapter")
        }
        model.load_state_dict(adapter_state_dict, strict=False)

    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu")

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        if "pt" in opt.resume[:-4]:
            if "asr" in opt.resume[:25]:
                model.load_state_dict(checkpoint["model"])
            else:
                for k, v in checkpoint["state_dict"].items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # model.load_state_dict(checkpoint["model"])
                model.load_state_dict(new_state_dict)
        else:
            # model.load_state_dict(checkpoint["state_dict"])
            model.load_state_dict(checkpoint["model"], strict=True)
        if opt.resume_all:
            opt.start_epoch = checkpoint["epoch"] + 1
    else:
        logger.warning(
            "If you intend to evaluate the model, please specify --resume with ckpt path"
        )

    return model


def start_inference(train_opt=None, split=None, splitfile=None):
    if train_opt is not None:
        opt = TestOptions().parse(train_opt.a_feat_dir)
    else:
        opt = TestOptions().parse()
    if split is not None:
        opt.eval_split_name = split
    if splitfile is not None:
        opt.eval_path = splitfile

    opt.cfg = nncore.Config.from_file(opt.config)

    print(opt.eval_split_name)
    print(opt.eval_path)
    logger.info("Setup config, data and model...")

    cudnn.benchmark = True
    cudnn.deterministic = False

    assert opt.eval_path is not None
    if opt.eval_split_name == "val":
        loadlabel = True
    else:
        loadlabel = False

    eval_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type=opt.q_feat_type,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=loadlabel,  # opt.eval_split_name == "val",
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=opt.dset_domain,
    )
    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory,
    )

    model = setup_model(opt)
    # save_submission_filename = "hl_{}_submission.jsonl".format(opt.eval_split_name)

    logger.info("Starting inference...")

    with torch.no_grad():
        logger.info("Generate submissions")
        model.eval()
        submissions = compute_mr_results(model, eval_loader, opt)
    return submissions


from sys import argv

if __name__ == "__main__":
    # split, splitfile = argv
    _, _, _, _, _, split, _, splitfile = argv

    print(f"Running inference with split: {split}, splitfile: {splitfile}")
    submissions = start_inference(split=split, splitfile=splitfile)
    # print(submissions)
