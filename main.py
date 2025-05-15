import numpy as np
from scipy.ndimage import distance_transform_edt, zoom
from skimage import measure
from PIL import Image, ImageOps, ImageFilter
import math
import argparse
import numba
from numba import njit, prange
import cv2


def smooth_normals(img, d=20, sigmaColor=35):
    img_np = np.array(img)
    smoothed = cv2.bilateralFilter(img_np, d, sigmaColor, 75)
    return Image.fromarray(smoothed)


def upscale(img, upscale_factor):
    # Interpolate to higher resolution (bicubic)
    # img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.array(img)
    img_np = zoom(img_np, upscale_factor, order=3)
    # img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image-file", help="Path to the input image", required=True
    )
    parser.add_argument(
        "-o", "--output-file", help="Path to the output normal map", required=True
    )
    parser.add_argument(
        "-u",
        "--upscale-factor",
        type=float,
        help="How much to upscale the image prior to normal generation",
    )
    parser.add_argument(
        "-t", "--threshold", help="Threshold for mask after upscaling", default=40
    )
    parser.add_argument(
        "-s",
        "--tangent-sample-size",
        help="How far forward and backward to look on contour for tangent estimation",
    )
    args = parser.parse_args()

    # Load a black-on-white image of the text
    img = Image.open(args.image_file).convert("L")

    if args.upscale_factor:
        print("Upscaling image")
        img = upscale(img, args.upscale_factor)

    img = ImageOps.invert(img)

    # print("Upscaling and masking")
    # mask = upscale_and_mask(img, args.upscale_factor, args.threshold)

    mask = np.array(img) > args.threshold

    print("Computing normals")
    normal_map = compute_contour_based_normals(np.array(mask), strength=1.0)

    print("Smoothing normals")
    normal_map = smooth_normals(normal_map)

    print("Saving output")
    normal_map.save(args.output_file)


@njit
def normalize(vx, vy):
    norm = math.sqrt(vx * vx + vy * vy) + 1e-8
    return vx / norm, vy / norm


@njit(parallel=True)
def compute_normals(
    mask,
    contour_x,
    contour_y,
    contour_id_map,
    contour_index_map,
    contours_flat,
    contour_start_idx,
    nx2d,
    ny2d,
    tangent_sample_size=4,
):
    h, w = mask.shape
    for y in prange(h):
        for x in range(w):
            if not mask[y, x]:
                continue

            cy = contour_y[y, x]
            cx = contour_x[y, x]
            c_id = contour_id_map[cy, cx]
            c_idx = contour_index_map[cy, cx]

            if c_id < 0 or c_idx < 0:
                continue

            start = contour_start_idx[c_id]
            end = contour_start_idx[c_id + 1]
            contour = contours_flat[start:end]
            n = end - start

            i0 = max(c_idx - tangent_sample_size, 0)
            i1 = min(c_idx + tangent_sample_size, n - 1)

            y0, x0 = contour[i0]
            y2, x2 = contour[i1]

            # Vector from contour to pixel
            vx, vy = x - cx, y - cy
            v0x, v0y = normalize(vx, vy)

            # Neighbor vectors
            v1x, v1y = normalize(x0 - cx, y0 - cy)
            v2x, v2y = normalize(x2 - cx, y2 - cy)

            dot1 = max(-1.0, min(v0x * v1x + v0y * v1y, 1.0))
            dot2 = max(-1.0, min(v0x * v2x + v0y * v2y, 1.0))
            a1 = math.acos(dot1)
            a2 = math.acos(dot2)

            # Tangents
            t1x, t1y = normalize(cx - x0, cy - y0)
            t2x, t2y = normalize(x2 - cx, y2 - cy)

            # Rotate clockwise to get normals
            n1x, n1y = t1y, -t1x
            n2x, n2y = t2y, -t2x

            if a1 < a2:
                nx2d[y, x], ny2d[y, x] = n1x, n1y
            else:
                nx2d[y, x], ny2d[y, x] = n2x, n2y


def compute_contour_based_normals(mask, strength=1.0):
    h, w = mask.shape

    # --- Extract contours ---
    contours = measure.find_contours(mask, 0.01)
    contour_img = np.zeros((h, w), dtype=bool)
    contour_index_map = -np.ones((h, w), dtype=np.int32)
    contour_id_map = -np.ones((h, w), dtype=np.int32)

    contour_flat_list = []
    contour_start_idx = [0]

    for contour_id, contour in enumerate(contours):
        for i, (y, x) in enumerate(contour):
            ry, rx = int(round(y)), int(round(x))
            if 0 <= ry < h and 0 <= rx < w:
                contour_img[ry, rx] = True
                contour_index_map[ry, rx] = i
                contour_id_map[ry, rx] = contour_id
        contour_flat_list.extend(contour)
        contour_start_idx.append(len(contour_flat_list))

    contours_flat = np.array(contour_flat_list, dtype=np.float32)
    contour_start_idx = np.array(contour_start_idx, dtype=np.int32)

    # --- Distance transform ---
    _, indices = distance_transform_edt(~contour_img, return_indices=True)
    contour_y, contour_x = indices

    nx2d = np.zeros((h, w), dtype=np.float32)
    ny2d = np.zeros((h, w), dtype=np.float32)

    compute_normals(
        mask,
        contour_x,
        contour_y,
        contour_id_map,
        contour_index_map,
        contours_flat,
        contour_start_idx,
        nx2d,
        ny2d,
        8,
    )

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
    normal_rgb[~mask] = [128, 128, 255]

    return Image.fromarray(normal_rgb, mode="RGB")


if __name__ == "__main__":
    main()
