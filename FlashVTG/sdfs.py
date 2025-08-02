import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    ShortSideScale,
)

# 사전학습 SlowFast 모델 로드
model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


# SlowFast 유입을 위한 변환(아래 PackPathway는 필수)
class PackPathway(torch.nn.Module):
    def forward(self, frames):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // 4).long(),
        )
        return [slow_pathway, fast_pathway]


transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(32),  # 2초당 32프레임 등
            Lambda(lambda x: x / 255.0),
            NormalizeVideo([0.45] * 3, [0.225] * 3),
            ShortSideScale(size=256),
            CenterCropVideo(256),
            PackPathway(),
        ]
    ),
)

# 비디오 읽고 원하는 2초 구간 추출
video = EncodedVideo.from_path("/hub_data2/intern/jinwoo/qvhighlight_examples/bP5KfdFJzC4_60.0_210.0.mp4")
video_data = video.get_clip(start_sec=0, end_sec=2)
video_data = transform(video_data)
inputs = [i.to(device)[None, ...] for i in video_data["video"]]

# 추론 (bottleneck feature는 hook 필요)
with torch.no_grad():
    feat = model(inputs)  # (기본 output은 logits)

# bottleneck 2304 feature 추출법은 아래와 같이 hook 사용
features = {}


def hook_fn(module, input, output):
    features["bottleneck"] = output.detach()


handle = model.blocks[-1].proj.register_forward_hook(hook_fn)
with torch.no_grad():
    _ = model(inputs)
bottleneck_feature = features["bottleneck"].reshape(1, -1)  # [1, 2304] 등
print(f"Bottleneck feature shape: {bottleneck_feature.shape}")
handle.remove()
# import torch
# import clip
# from PIL import Image

# clip_model, preprocess = clip.load("ViT-B/32", device)
# frame = Image.open("sample_frame.jpg")
# image_input = preprocess(frame).unsqueeze(0).to(device)

# with torch.no_grad():
#     image_features = clip_model.encode_image(image_input)
# image_features = image_features / image_features.norm(dim=-1, keepdim=True)
# import numpy as np

# concat_feature = np.concatenate(
#     [bottleneck_feature.cpu().numpy(), image_features.cpu().numpy()], axis=-1
# )  # shape: [1, 2816]
