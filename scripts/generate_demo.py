import cv2
import numpy as np

def generate_demo_video(output_path="demo.mp4", duration_sec=5, fps=30):
    """
    Generates a synthetic demo video if you don't have one.
    You already have a demo.mp4 (Earth at night), so you don't strictly need to run this.
    """
    print(f"Generating synthetic {output_path}...")
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames = duration_sec * fps
    for i in range(frames):
        # Create a moving gradient pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        color_intensity = int((np.sin(i / 10.0) + 1) * 127)
        frame[:, :] = (color_intensity, color_intensity, color_intensity)
        
        cv2.putText(frame, f"Frame: {i}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(frame)

    out.release()
    print("Done!")

if __name__ == "__main__":
    generate_demo_video()
