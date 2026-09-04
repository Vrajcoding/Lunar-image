"""
create_sample_images.py
Generate 3 synthetic lunar image pairs for testing:
  Pair 1 — Easy:   8° rotation + small translation (expected: RMSE < 1.0)
  Pair 2 — Medium: 20° rotation + scale 1.1
  Pair 3 — Same:   identical images (perfect-match baseline)
"""
import os
import cv2
import numpy as np



def make_lunar_base(seed: int = 42, size: int = 512) -> np.ndarray:
    """Generate a synthetic lunar surface with craters, ridges, and texture."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 110, dtype=np.uint8)
    noise = rng.normal(0, 12, (size, size)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    craters = [
        (150, 150, 45), (320, 180, 65), (200, 360, 55),
        (390, 390, 38), (100, 420, 28), (430, 110, 50),
        (260, 260, 30), (80,  200, 20), (450, 300, 42),
    ]
    for (cx, cy, r) in craters:
        if cx + r < size and cy + r < size:
            cv2.circle(img, (cx, cy), r, (65,), -1)
            cv2.circle(img, (cx + 4, cy + 4), max(r - 5, 2), (155,), -1)
            cv2.circle(img, (cx, cy), r, (25,), 2)

    # Add ridges for extra texture/features
    for i in range(0, size, 60):
        cv2.line(img, (i, 0), (i + 30, size), (90,), 1)

    return img


def generate_lunar_sample_images():
    out_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    os.makedirs(out_dir, exist_ok=True)

    base = make_lunar_base(seed=42, size=512)
    center = (256, 256)

    # ── Pair 1: Easy — 8° rotation + minor translation ────────────────────
    ref1_path = os.path.join(out_dir, "ref1.png")
    src1_path = os.path.join(out_dir, "source1.png")
    cv2.imwrite(ref1_path, base)
    M1 = cv2.getRotationMatrix2D(center, 8, 1.02)
    M1[0, 2] += 12
    M1[1, 2] -= 8
    cv2.imwrite(src1_path, cv2.warpAffine(base, M1, (512, 512)))
    print(f"[Pair 1 — Easy]   ref: {ref1_path}")
    print(f"                  src: {src1_path}")

    # ── Pair 2: Medium — 20° rotation + scale 1.1 ──────────────────────────
    ref2_path = os.path.join(out_dir, "ref2.png")
    src2_path = os.path.join(out_dir, "source2.png")
    cv2.imwrite(ref2_path, base)
    M2 = cv2.getRotationMatrix2D(center, 20, 1.1)
    cv2.imwrite(src2_path, cv2.warpAffine(base, M2, (512, 512)))
    print(f"[Pair 2 — Medium] ref: {ref2_path}")
    print(f"                  src: {src2_path}")

    # ── Pair 3: Same — identical (perfect-match baseline) ───────────────────
    same_path = os.path.join(out_dir, "same.png")
    cv2.imwrite(same_path, base)
    print(f"[Pair 3 — Same]   path: {same_path}")

    print("\nDone. Use these images with: curl -X POST http://localhost:8001/register \\")
    print("  -F 'source=@tests/sample_images/source1.png' \\")
    print("  -F 'reference=@tests/sample_images/ref1.png'")


if __name__ == "__main__":
    generate_lunar_sample_images()
