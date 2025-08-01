import os
import argparse
import torch
from torchvision.io import read_video
from torchvision.models.video import r3d_18
from torchvision import transforms
from transformers import BertTokenizer, BertModel
import json
import logging
import sys


sys.path.append("/home/intern/jinwoo/FlashVTG")
logger = logging.getLogger(__name__)


def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    from FlashVTG.model import build_model1

    model, criterion = build_model1(opt)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "lr": opt.lr,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop, gamma=0.5)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-4)

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
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            opt.start_epoch = checkpoint["epoch"] + 1
    else:
        logger.warning(
            "If you intend to evaluate the model, please specify --resume with ckpt path"
        )

    return model, criterion, optimizer, lr_scheduler


# 영상 feature 추출 함수 (개선된 버전)
def extract_video_feature(video_path, device="cpu"):
    try:
        # pts_unit='sec'로 설정하여 경고 해결
        video, _, info = read_video(video_path, pts_unit="sec")
        print(f"Video loaded: {video.shape}, FPS: {info.get('video_fps', 'unknown')}")

        # 영상이 너무 길면 처음 16프레임만 사용
        if video.shape[0] > 16:
            video = video[:16]
        elif video.shape[0] < 16:
            # 프레임이 부족하면 마지막 프레임을 반복
            last_frame = video[-1:]
            repeat_times = 16 - video.shape[0]
            video = torch.cat([video, last_frame.repeat(repeat_times, 1, 1, 1)], dim=0)

        # 정규화 및 차원 변환
        video = video.float() / 255.0
        video = video.permute(0, 3, 1, 2)  # (T, C, H, W)
        video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)

        # r3d_18 모델로 feature 추출
        model = r3d_18(pretrained=True).to(device)
        model.eval()
        with torch.no_grad():
            feat = model(video.to(device))
        return feat.flatten().cpu().numpy()
    except Exception as e:
        print(f"Error extracting video feature: {e}")
        # 에러 시 더미 feature 반환
        return torch.zeros(512).numpy()


# 쿼리 feature 추출 함수 (개선된 버전)
def extract_text_feature(query, device="cpu"):
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
        model.eval()
        inputs = tokenizer(
            query, return_tensors="pt", max_length=512, truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in inputs.items()})
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    except Exception as e:
        print(f"Error extracting text feature: {e}")
        # 에러 시 더미 feature 반환
        return torch.zeros(768).numpy()


# 영상 duration 추출 함수 (개선된 버전)
def extract_duration(video_path):
    try:
        _, _, info = read_video(video_path, pts_unit="sec")
        if "duration" in info:
            return float(info["duration"])
        elif "video_fps" in info and "video_nframes" in info:
            return float(info["video_nframes"]) / float(info["video_fps"])
        else:
            print("Warning: Could not extract duration, using default 100.0")
            return 100.0
    except Exception as e:
        print(f"Error extracting duration: {e}")
        return 100.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="영상 경로")
    parser.add_argument("--query", type=str, required=True, help="쿼리 문장")
    args = parser.parse_args()

    # config, resume, device 자동 지정
    config_path = "data/MR.py"
    resume_path = "results/QVHighlights_SF+Clip/model_best.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Video path: {args.video_path}")
    print(f"Query: {args.query}")

    # duration 자동 추출
    duration = extract_duration(args.video_path)
    print(f"Video duration: {duration:.2f} seconds")

    # 실시간 feature 추출
    print("Extracting video feature...")
    video_feat = extract_video_feature(args.video_path, device=device)
    print(f"Video feature shape: {video_feat.shape}")

    print("Extracting text feature...")
    text_feat = extract_text_feature(args.query, device=device)
    print(f"Text feature shape: {text_feat.shape}")

    # FlashVTG 모델 옵션 세팅
    from FlashVTG.config import TestOptions
    import nncore

    opt = TestOptions().parse()
    opt.config = config_path
    opt.resume = resume_path
    opt.device = torch.device(device)
    opt.cfg = nncore.Config.from_file(opt.config)
    opt.eval_split_name = "test"
    opt.results_dir = "./results"

    # 모델 로드
    print("Loading model...")
    model, criterion, _, _ = setup_model(opt)
    model.eval()

    # FlashVTG 모델 입력 포맷에 맞게 가공
    print("Running inference...")
    with torch.no_grad():
        model_inputs = {
            "video_feat": torch.tensor(video_feat).unsqueeze(0).to(device),
            "query_feat": torch.tensor(text_feat).unsqueeze(0).to(device),
            "duration": torch.tensor([duration]).to(device),
        }
        outputs = model(**model_inputs)

    # 결과 출력
    print("\n=== 예측 결과 ===")
    print(json.dumps(outputs, indent=2, ensure_ascii=False, default=str))
