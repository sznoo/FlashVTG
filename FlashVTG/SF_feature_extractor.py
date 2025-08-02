import torch
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
import os

# 모델 임포트 및 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
model = model.eval().to(device)

# 변환 파이프라인 정의
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 30
sampling_rate = 2
frames_per_second = 30
alpha = 4  # slowfast 알파 설정


class PackPathway(torch.nn.Module):
    """SlowFast 입력 포맷을 위한 변환 모듈."""

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // alpha).long(),
        )
        return [slow_pathway, fast_pathway]


transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size),
            PackPathway(),
        ]
    ),
)

# 입력 비디오 세팅
video_path = "/hub_data2/intern/jinwoo/qvhighlight_examples/bP5KfdFJzC4_60.0_210.0.mp4"  # 대상 비디오의 경로
clip_duration = (num_frames * sampling_rate) / frames_per_second
start_sec = 0
end_sec = start_sec + clip_duration

# 비디오 로드 및 변환
video = EncodedVideo.from_path(video_path)
video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
video_data = transform(video_data)


bottleneck_features = {}


def hook_fn(module, input, output):
    bottleneck_features["feat"] = output.detach()


hook_handle = model.blocks[-2].register_forward_hook(hook_fn)

# 디바이스 올리기 및 배치 차원 추가
inputs = video_data["video"]
inputs = [i.to(device)[None, ...] for i in inputs]

# 추론 및 feature 추출
with torch.no_grad():
    feature = model(inputs)  # 기본적으로 (배치, 클래스수) 출력, 필요시 내부 임베딩 후킹
feat = bottleneck_features["feat"]
# feat = feat.reshape(feat.shape[0], -1)  # 필요시 2D로
print("Bottleneck feature shape:", feat.shape)  # 예: (1, 2304)

# hook 해제
hook_handle.remove()

# 원하는 경우 feature를 numpy로 저장
# video_feature_array = feature.cpu().numpy()
# print("Feature vector shape:", video_feature_array.shape)
