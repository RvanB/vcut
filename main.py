import numpy as np
from scipy.ndimage import distance_transform_edt, zoom, binary_closing
from skimage import measure
from PIL import Image, ImageOps, ImageFilter
import math
import argparse
from numba import njit, prange
import cv2
import os

def smooth_normals(img, d=35, sigmaColor=35):
    img_np = np.array(img)
    smoothed = cv2.bilateralFilter(img_np, d, sigmaColor, 50)
    return Image.fromarray(smoothed)


def upscale(img, upscale_factor):
    img_np = np.array(img)
    img_np = zoom(img_np, upscale_factor, order=3)
    return Image.fromarray(img_np)


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


@njit
def estimate_tangent(
    before_contour_point, closest_contour_point, after_contour_point, point
):
    # Vectors
    contour_to_point = normalize(point - closest_contour_point)
    before_vector = normalize(before_contour_point - closest_contour_point)
    after_vector = normalize(after_contour_point - closest_contour_point)

    angle_to_before = calculate_angle(contour_to_point, before_vector)
    angle_to_after = calculate_angle(contour_to_point, after_vector)

    if angle_to_before < angle_to_after:
        tangent = normalize(closest_contour_point - before_contour_point)
    else:
        tangent = normalize(after_contour_point - closest_contour_point)

    return tangent


@njit(parallel=True)
def process(
        *,
        mask,
        contourX,
        contourY,
        contour_id_map,
        contour_index_map,
        contours_flat,
        contour_start_idx,
        normalX,
        normalY,
        displacement,
        tangent_sample_size
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

            # Indices in contour before and after closest point
            i0 = max(c_idx - tangent_sample_size, 0)
            i1 = min(c_idx + tangent_sample_size, n - 1)

            # Get before and after point locations
            before_contour_point = np.array(list(contour[i0]))
            after_contour_point = np.array(list(contour[i1]))

            tangent = estimate_tangent(
                before_contour_point, closest_contour_point, after_contour_point, point
            )
            normal = -tangent[1], tangent[0]
            normalX[y, x], normalY[y, x] = normal

            # Project onto tangent
            a = point - closest_contour_point
            proj = np.dot(a, tangent) * tangent + closest_contour_point
            
            distance = np.linalg.norm(point - proj)
            displacement[y, x] = distance
            


def generate_textures(mask, tangent_sample_size, strength):
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

    displacement = np.zeros((h, w), dtype=np.float32)

    # Populate normalX and normalY
    process(
        mask=mask,
        contourX=contourX,
        contourY=contourY,
        contour_id_map=contour_id_map,
        contour_index_map=contour_index_map,
        contours_flat=contours_flat,
        contour_start_idx=contour_start_idx,
        normalX=normalX,
        normalY=normalY,
        displacement=displacement,
        tangent_sample_size=tangent_sample_size
    )

    # Post-process displacement image
    displacement = -displacement
    displacement[~mask] = 0
    displacement -= np.min(displacement)
    displacement /= np.max(displacement)
    displacement = (displacement * 255).astype(np.uint8)

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
    
    return Image.fromarray(normal_rgb, mode="RGB"), Image.fromarray(displacement)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image-file", help="Path to the input image", required=True
    )
    parser.add_argument(
        "-u",
        "--upscale-factor",
        type=float,
        help="How much to upscale the image prior to normal generation"
    )
    parser.add_argument(
        "-T", "--threshold", help="Threshold for mask after upscaling", default=40
    )
    parser.add_argument(
        "-t",
        "--tangent-sample-size",
        type=int,
        help="How far forward and backward to look on contour for tangent estimation",
        default=4
    )
    parser.add_argument(
        "-s",
        "--strength",
        type=float,
        help="Strength of the generated normal",
        default=1.0
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Load a black-on-white image of the text
    img = Image.open(args.image_file).convert("L")

    if args.upscale_factor:
        print("Upscaling image")
        img = upscale(img, args.upscale_factor)

    img = ImageOps.invert(img)

    mask = (np.array(img) > args.threshold).astype(np.uint8)

    mask = binary_closing(mask, iterations=1)

    print("Computing textures")
    normal_map, displacement_map = generate_textures(
        mask=np.array(mask),
        tangent_sample_size=args.tangent_sample_size,
        strength=args.strength)

    print("Smoothing normals")

    input_basename, ext = os.path.splitext(args.image_file)

    normal_outpath = f"{input_basename}_normal.png"
    displacement_outpath = f"{input_basename}_displacement.png"
    
    normal_map = smooth_normals(normal_map)

    print("Saving outputs")
    normal_map.save(normal_outpath)
    displacement_map.save(displacement_outpath)


if __name__ == "__main__":
    main()
