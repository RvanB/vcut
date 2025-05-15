import numpy as np
from scipy.ndimage import distance_transform_edt, zoom
from skimage import measure
from PIL import Image, ImageOps, ImageFilter
import math
import argparse
from numba import njit, prange
import cv2


def smooth_normals(img, d=20, sigmaColor=35):
    img_np = np.array(img)
    smoothed = cv2.bilateralFilter(img_np, d, sigmaColor, 50)
    return Image.fromarray(smoothed)


def upscale(img, upscale_factor):
    img_np = np.array(img)
    img_np = zoom(img_np, upscale_factor, order=3)
    return Image.fromarray(img_np)


def parse_args():
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
    return args


@njit
def calculate_angle(v0, v1):
    dot = np.dot(v0, v1)
    return math.acos(dot)


@njit
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


@njit(parallel=True)
def compute_normals(
    mask,
    contourX,
    contourY,
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

            point = np.array([y, x]).astype(np.float32)

            closest_contour_point = np.array([contourY[y, x], contourX[y, x]])

            # ID of closest contour pixel
            c_id = contour_id_map[closest_contour_point[0], closest_contour_point[1]]
            # index of pixel in the contour
            c_idx = contour_index_map[
                closest_contour_point[0], closest_contour_point[1]
            ]

            if c_id < 0 or c_idx < 0:
                continue

            # Get the entire contour
            start = contour_start_idx[c_id]
            end = contour_start_idx[c_id + 1]
            contour = contours_flat[start:end]
            n = end - start

            contour_to_point = normalize(point - closest_contour_point)

            # Indices in contour before and after closest point
            i0 = max(c_idx - tangent_sample_size, 0)
            i1 = min(c_idx + tangent_sample_size, n - 1)

            # Get before and after point locations
            before_point = np.array(list(contour[i0]))
            after_point = np.array(list(contour[i1]))

            # Neighbor vectors
            before_vector = normalize(before_point - closest_contour_point)
            after_vector = normalize(after_point - closest_contour_point)
            # bvx, bvy = normalize(bvx - cx, bvy - cy)
            # v2x, v2y = normalize(x2 - cx, y2 - cy)

            angle_to_before = calculate_angle(contour_to_point, before_vector)
            angle_to_after = calculate_angle(contour_to_point, after_vector)

            if angle_to_before < angle_to_after:
                tangent = normalize(closest_contour_point - before_point)
            else:
                tangent = normalize(after_point - closest_contour_point)

            nx2d[y, x], ny2d[y, x] = -tangent[1], tangent[0]


def compute_contour_based_normals(mask, strength=1.0):
    h, w = mask.shape

    contours = measure.find_contours(mask, 0.01)

    # Contour bitmap
    contour_img = np.zeros((h, w), dtype=bool)

    # index of pixel in contour (not globally unique)
    contour_index_map = -np.ones((h, w), dtype=np.int32)

    # IDs of contour (not pixels)
    contour_id_map = -np.ones((h, w), dtype=np.int32)

    # List of all points, across contours
    contour_flat_list = []

    # Indices of breaks between contours
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

    # Calculate distances to contour throughout image
    _, indices = distance_transform_edt(~contour_img, return_indices=True)
    contourY, contourX = indices

    normalX = np.zeros((h, w), dtype=np.float32)
    normalY = np.zeros((h, w), dtype=np.float32)

    # Populate normalX and normalY
    compute_normals(
        mask,
        contourX,
        contourY,
        contour_id_map,
        contour_index_map,
        contours_flat,
        contour_start_idx,
        normalX,
        normalY,
        8,
    )

    normalZ = np.full_like(normalX, 1.0 / strength)
    length = np.sqrt(normalX**2 + normalY**2 + normalZ**2)
    nx = normalX / length
    ny = normalY / length
    nz = normalZ / length

    normal_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    normal_rgb[..., 0] = ((nx + 1) * 127.5).astype(np.uint8)
    normal_rgb[..., 1] = ((ny + 1) * 127.5).astype(np.uint8)
    normal_rgb[..., 2] = ((nz + 1) * 127.5).astype(np.uint8)
    normal_rgb[~mask] = [128, 128, 255]

    return Image.fromarray(normal_rgb, mode="RGB")


def main():
    args = parse_args()

    # Load a black-on-white image of the text
    img = Image.open(args.image_file).convert("L")

    if args.upscale_factor:
        print("Upscaling image")
        img = upscale(img, args.upscale_factor)

    img = ImageOps.invert(img)

    mask = np.array(img) > args.threshold

    print("Computing normals")
    normal_map = compute_contour_based_normals(np.array(mask), strength=1.0)

    print("Smoothing normals")
    normal_map = smooth_normals(normal_map)

    print("Saving output")
    normal_map.save(args.output_file)


if __name__ == "__main__":
    main()
