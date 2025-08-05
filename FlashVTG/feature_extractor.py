import numpy as np
from numpy.linalg import norm


def inspect_npz(path):
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


def cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (norm(a_flat) * norm(b_flat) + 1e-8)


def compare_npz_files(path1, path2, n_rows=75, verbose=False):
    try:
        data1 = np.load(path1)
        data2 = np.load(path2)

        keys1 = set(data1.files)
        keys2 = set(data2.files)

        if keys1 != keys2:
            print("❌ 키 불일치:")
            print(f"  - {path1}: {keys1}")
            print(f"  - {path2}: {keys2}")
            return

        for key in keys1:
            print(f"\n🔍 비교 중: key = '{key}'")

            arr1 = data1[key]
            arr2 = data2[key]

            if arr1.shape != arr2.shape:
                print(f"❌ shape 불일치: {arr1.shape} vs {arr2.shape}")
                min_len = min(arr1.shape[0], arr2.shape[0])
                arr1 = arr1[:min_len]
                arr2 = arr2[:min_len]
                print(f"↪️  shape 맞춰 비교: ({arr1.shape})")

            if arr1.dtype != arr2.dtype:
                print(f"❌ dtype 불일치: {arr1.dtype} vs {arr2.dtype}")

            # Cosine similarity over flattened vectors
            cos_sim = cosine_similarity(arr1, arr2)

            # Diff stats
            diffs = np.abs(arr1 - arr2)
            max_diff = np.max(diffs)
            mean_diff = np.mean(diffs)
            row_means = np.mean(diffs, axis=1) if len(diffs.shape) == 2 else []

            print(f"✅ shape: {arr1.shape}")
            print(f"   - Cosine Similarity (flattened): {cos_sim:.6f}")
            print(f"   - Mean Absolute Diff: {mean_diff:.6f}")
            print(f"   - Max Diff: {max_diff:.6f}")

            if len(row_means) > 0:
                print(
                    f"   - Row-wise mean diff (top 5 rows): {[round(f, 6) for f in row_means[:5]]}"
                )
                significant_rows = np.where(row_means > 1e-3)[0]
                if len(significant_rows) > 0:
                    print(f"   - ⚠️ diff > 1e-3 at rows: {significant_rows.tolist()}")

            # Optional: compare EOS row directly if token-level output
            if len(arr1.shape) == 2 and arr1.shape[1] == 512:
                eos_cos = cosine_similarity(arr1[-1], arr2[-1])
                eos_diff = np.abs(arr1[-1] - arr2[-1])
                print(f"   - EOS row Cosine Similarity: {eos_cos:.6f}")
                print(f"   - EOS Max Diff: {np.max(eos_diff):.6f}")

            if verbose:
                print("   - Sample arr1 row 0:", arr1[0])
                print("   - Sample arr2 row 0:", arr2[0])
                print("   - Diff row 0:", diffs[0])

    except Exception as e:
        print(f"⚠️ 비교 실패: {e}")


if __name__ == "__main__":
    vid = "RoripwjYFp8_60.0_210.0"
    ftype = "slowfast_features"
    path1 = f"/hub_data2/intern/jinwoo/qvhighlight/{ftype}/{vid}.npz"
    path2 = f"/hub_data2/intern/jinwoo/qvhl_videos/features/{vid}.npz"
    # compare_npz_files(path1, path2, n_rows=75, verbose=True)

    # p1 = "/hub_data2/intern/jinwoo/qvhl_videos/clip_vit_features/RoripwjYFp8_60.0_210.0.npz"
    # inspect_npz(p1)

    # p2 = "/hub_data2/intern/jinwoo/qvhighlight/clip_features/RoripwjYFp8_60.0_210.0.npz"
    # inspect_npz(p2)
    # compare_npz_files(p1, p2, n_rows=75, verbose=True)
    id = 781
    p3 = f"/hub_data2/intern/jinwoo/qvhl_videos/clip_text_features/qid{id}.npz"
    # "An Asian woman wearing a Boston t-shirt is in her home talking."
    inspect_npz(p3)
    p4 = f"/hub_data2/intern/jinwoo/qvhighlight/clip_text_features/qid{id}.npz"
    # "An Asian woman wearing a Boston t-shirt is in her home talking."
    inspect_npz(p4)
    compare_npz_files(p3, p4, n_rows=77, verbose=False)
