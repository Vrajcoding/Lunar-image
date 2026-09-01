import os
import cv2
import numpy as np

def generate_lunar_sample_images():
    out_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    os.makedirs(out_dir, exist_ok=True)
    
    # Create base synthetic lunar terrain (500x500 grayscale image with craters)
    np.random.seed(42)
    img = np.full((500, 500), 120, dtype=np.uint8)
    
    # Add random noise/texture
    noise = np.random.normal(0, 15, (500, 500)).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Add craters (circles with shadows)
    craters = [
        (150, 150, 40), (320, 180, 60), (200, 350, 50), 
        (380, 380, 35), (100, 400, 25), (420, 100, 45)
    ]
    
    for (cx, cy, r) in craters:
        cv2.circle(img, (cx, cy), r, (70,), -1)
        cv2.circle(img, (cx + 3, cy + 3), r - 4, (160,), -1)
        cv2.circle(img, (cx, cy), r, (30,), 2)
        
    ref_path = os.path.join(out_dir, "ref1.png")
    cv2.imwrite(ref_path, img)
    
    # Create source image with affine transform (rotation 8 deg, scale 1.05, translation +10, -15)
    M = cv2.getRotationMatrix2D((250, 250), 8, 1.02)
    M[0, 2] += 12
    M[1, 2] -= 8
    src_img = cv2.warpAffine(img, M, (500, 500))
    
    src_path = os.path.join(out_dir, "source1.png")
    cv2.imwrite(src_path, src_img)
    print(f"Generated sample images:\n  Reference: {ref_path}\n  Source: {src_path}")

if __name__ == "__main__":
    generate_lunar_sample_images()
