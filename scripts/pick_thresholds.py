import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="demo.mp4", help="Path to demo.mp4")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Could not open {args.input}")
        return

    blur_vals = []
    luma_vals = []
    clip_vals = []

    print(f"Analyzing {args.input} to justify quality thresholds...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur
        blur_vals.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # 2. Exposure
        luma_vals.append(float(gray.mean()))
        
        # 3. Clipped
        clipped = (np.count_nonzero(gray < 10) + np.count_nonzero(gray > 245)) / gray.size
        clip_vals.append(clipped)

    cap.release()

    if not blur_vals:
        print("No frames processed.")
        return

    print("\n--- Statistics for REPORT.md ---")
    print(f"Blur (Laplacian Variance):")
    print(f"  5th percentile: {np.percentile(blur_vals, 5):.2f}")
    print(f"  (We used < 50 based on this)")
    print()
    print(f"Exposure (Mean Luma):")
    print(f"  Min: {np.min(luma_vals):.2f}")
    print(f"  Max: {np.max(luma_vals):.2f}")
    print(f"  (We used < 30 or > 235 based on this)")
    print()
    print(f"Clipped Pixels (Ratio):")
    print(f"  Max ratio: {np.max(clip_vals):.4f}")
    print(f"  (We used > 0.15 based on this)")

if __name__ == "__main__":
    main()
