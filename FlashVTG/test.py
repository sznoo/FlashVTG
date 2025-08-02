import torch

# SlowFast R-50 모델 로드
model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
model.eval()

# feature를 저장할 전역 변수
bottleneck_features = {}


# hook 함수 정의
def hook_fn(module, input, output):
    bottleneck_features["feat"] = output.detach()


hook_handle = model.blocks[-2].register_forward_hook(hook_fn)
# (다를 경우 head.proj.register_forward_hook도 가능)

# 비디오 입력 구성 (예시: SlowFast 입력 파이프라인 참고)
dummy_input = [torch.randn(1, 3, 8, 224, 224), torch.randn(1, 3, 32, 224, 224)]

# 추론 실행 (여기서 hook이 발동하여 feat에 저장)
with torch.no_grad():
    _ = model(dummy_input)

# bottleneck feature 꺼내기 (예: [batch, 2304, 1, 1, 1] 또는 [batch, 2304]로 펼침)
feat = bottleneck_features["feat"]
feat = feat.reshape(feat.shape[0], -1)  # 필요시 2D로

print("Bottleneck feature shape:", feat.shape)  # 예: (1, 2304)

# hook 해제
hook_handle.remove()
