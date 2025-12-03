import numpy as np
from PIL import Image

# Load annotation
anno = np.load('/mnt/data/vivek/point_odyssey_v1.2/train/ani2/anno.npz')

# Sample frame dims
frame_path = '/mnt/data/vivek/point_odyssey_v1.2/train/ani2/frames/frame_000000.jpg'
with Image.open(frame_path) as img:
    W, H = img.size

print("Frame width:", W)
print("Frame height:", H)
print("NPZ Keys:", anno.files)

for frame_idx in range(len(anno['valids'])):

    print("\nIteration:", frame_idx)

    # ========== INTRINSICS ==========
    K = anno['intrinsics'][frame_idx]   # (3, 3)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # ========== EXTRINSICS ==========
    E = anno['extrinsics'][frame_idx]   # (4, 4)
    print("Extrinsic matrix shape:", E.shape)

    for point_idx in range(len(anno['valids'][frame_idx])):

        if not anno['visibs'][frame_idx][point_idx]:
            continue

        # Annotated 2D
        x2d, y2d = anno['trajs_2d'][frame_idx][point_idx]

        # World 3D
        Xw, Yw, Zw = anno['trajs_3d'][frame_idx][point_idx]

        # Convert to homogeneous coords
        Pw = np.array([Xw, Yw, Zw, 1.0])

        # World → Camera coordinates
        Pc = E @ Pw
        Xc, Yc, Zc = Pc[:3]

        if Zc <= 0:
            continue

        # Project onto image plane
        u = fx * (Xc / Zc) + cx
        v = fy * (Yc / Zc) + cy

        print("\nFrame:", frame_idx)
        print("Point:", point_idx)
        print("2D pixel (anno):", x2d, y2d)
        print("3D world:", [Xw, Yw, Zw])
        print("3D camera:", [Xc, Yc, Zc])
        print("3D → 2D (calc):", u, v)
        print("Diff:", (x2d - u, y2d - v))

        exit(0)
