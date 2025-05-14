import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage import measure
from PIL import ImageOps
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage import measure

def main():
    # Load a black-on-white image of the text
    img = Image.open("test-zoomed.png").convert("L")
    img = ImageOps.invert(img)

    mask = (np.array(img) > 40)

    normal_map = compute_contour_based_normals(mask, strength=1.0)
    normal_map.save("test-zoomed-norm.png")



def compute_contour_based_normals(mask, strength=1.0):
    h, w = mask.shape

    # --- Extract contours ---
    contours = measure.find_contours(mask, 0.5)
    contour_img = np.zeros((h, w), dtype=bool)
    contour_index_map = -np.ones((h, w), dtype=np.int32)
    contour_id_map = -np.ones((h, w), dtype=np.int32)

    contour_lookup = {}
    contour_id = 0
    for contour in contours:
        n = len(contour)
        for i, (y, x) in enumerate(contour):
            ry, rx = int(round(y)), int(round(x))
            if 0 <= ry < h and 0 <= rx < w:
                contour_img[ry, rx] = True
                contour_index_map[ry, rx] = i
                contour_id_map[ry, rx] = contour_id
                contour_lookup[(contour_id, i)] = (y, x)
        contour_id += 1

    # --- Distance transform ---
    dist, indices = distance_transform_edt(~contour_img, return_indices=True)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    contour_y = indices[0]
    contour_x = indices[1]

    nx2d = np.zeros((h, w), dtype=np.float32)
    ny2d = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue

            cy, cx = contour_y[y, x], contour_x[y, x]
            c_id = contour_id_map[cy, cx]
            c_idx = contour_index_map[cy, cx]

            if c_id < 0 or c_idx < 0:
                continue

            contour = contours[c_id]
            n = len(contour)

            i = c_idx
            prev_i = max(i - 3, 0)
            next_i = min(i + 3, n - 1) # Combine into a 3D vector and normalize

            # Edge segments
            y0, x0 = contour[prev_i]
            y1, x1 = contour[i]
            y2, x2 = contour[next_i]

            # Tangents
            t1 = np.array([x1 - x0, y1 - y0], dtype=np.float32)
            t2 = np.array([x2 - x1, y2 - y1], dtype=np.float32)

            t1_norm = np.linalg.norm(t1) + 1e-8
            t2_norm = np.linalg.norm(t2) + 1e-8
            t1 /= t1_norm
            t2 /= t2_norm

            # Normals (rotate CW)
            n1 = np.array([t1[1], -t1[0]])
            n2 = np.array([t2[1], -t2[0]])

            # Vector from contour to pixel
            inward = np.array([x - cx, y - cy], dtype=np.float32)
            magnitude = np.linalg.norm(inward)
            if magnitude > 2:
                inward /= np.linalg.norm(inward)
                d1 = abs(np.dot(n1, inward))
                d2 = abs(np.dot(n2, inward))

                chosen_n = n1 if d1 > d2 else n2

                # # Flip if wrong direction
                # if np.dot(chosen_n, inward) < 0:
                #     chosen_n = -chosen_n
            else:
                chosen_n = (n1 + n2) / 2.0

            nx2d[y, x] = chosen_n[0]
            ny2d[y, x] = chosen_n[1]

    # --- Construct 3D normal map ---
    nz = np.full_like(nx2d, 1.0 / strength)
    length = np.sqrt(nx2d**2 + ny2d**2 + nz**2)
    nx = nx2d / length
    ny = ny2d / length
    nz = nz / length

    normal_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    normal_rgb[..., 0] = ((nx + 1) * 127.5).astype(np.uint8)
    normal_rgb[..., 1] = ((ny + 1) * 127.5).astype(np.uint8)
    normal_rgb[..., 2] = ((nz + 1) * 127.5).astype(np.uint8)

    # Flat background
    normal_rgb[~mask] = [128, 128, 255]

    return Image.fromarray(normal_rgb, mode='RGB')


if __name__ == "__main__":
    main()
