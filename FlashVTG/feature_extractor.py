import numpy as np


def inspect_npz(path):
    """
    .npz 파일 내부의 키와 각 데이터 shape, dtype 등을 출력하는 함수

    Args:
        path (str): .npz 파일 경로
    """
    try:
        data = np.load(path)
        print(f"파일: {path}")
        print("포함된 키:", data.files)
        for key in data.files:
            value = data[key]
            print(f"  - key: '{key}' | shape: {value.shape}, dtype: {value.dtype}")
    except Exception as e:
        print(f"⚠️ 파일 로딩 실패: {e}")
    print("")


if __name__ == "__main__":
    video_path = "/hub_data2/intern/jinwoo/qvhighlight_examples/slowfast_features/bP5KfdFJzC4_60.0_210.0.npz"
    inspect_npz(video_path)
