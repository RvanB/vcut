import numpy as np
from PIL import Image
from skimage import measure
import sys

def image_to_mask(image_path):
    image = Image.open(image_path).convert('L')  # grayscale
    image.save("something.png")
    mask = np.array(image) < 255
    return mask.astype(np.uint8)

def compute_inward_normals(mask):    
    contours = measure.find_contours(mask, 0.5)
    normals = []

    for contour in contours:
        for i in range(1, len(contour) - 1):
            y0, x0 = contour[i-1]

            contour_img[y0, x0] = 1
            
            y1, x1 = contour[i+1]
            dx, dy = x1 - x0, y1 - y0
            length = np.hypot(dx, dy)
            if length == 0:
                continue
            # Normal pointing inward
            nx, ny = -dy / length, dx / length
            x, y = int(round(contour[i][1])), int(round(contour[i][0]))
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                normals.append((y, x, nx, ny))

    return normals

def simulate_vcut_zbuffer(mask, normals, depth_steps=10, slope=1.0):
    h, w = mask.shape
    zbuffer = np.zeros((h, w), dtype=np.float32)

    for y0, x0, nx, ny in normals:
        for d in range(depth_steps):
            # Calculate x, y coordinates of the inward-moving depth projection
            x = int(round(x0 + nx * d))
            y = int(round(y0 + ny * d))

            # Ensure the coordinates stay within image bounds and correspond to a carved area
            if 0 <= x < w and 0 <= y < h and mask[y, x]:
                # Calculate depth based on constant slope, no need to scale it
                depth = slope * d  # Constant slope depth calculation
                zbuffer[y, x] = max(zbuffer[y, x], depth)
                
    return zbuffer


def depth_to_normal_map(zbuffer, strength=1.0):
    dzdx = np.gradient(zbuffer, axis=1)
    dzdy = np.gradient(zbuffer, axis=0)
    # dz = 1.0 / strength
    dz = np.full_like(dzdx, 1.0 / strength)

    normals = np.stack((-dzdx, -dzdy, dz), axis=-1)
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.maximum(norm, 1e-8)
    normal_rgb = ((normals + 1) / 2.0 * 255).astype(np.uint8)
    return Image.fromarray(normal_rgb, mode='RGB')

def main(image_path, output_path='normal_map.png'):
    mask = image_to_mask(image_path)
    
    normals = compute_inward_normals(mask)
    zbuffer = simulate_vcut_zbuffer(mask, normals, depth_steps=30, slope=1.0)
    normal_map = depth_to_normal_map(zbuffer)
    normal_map.save(output_path)
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python vcut_normalmap.py input_image.png [output_normal.png]")
    else:
        input_image = sys.argv[1]
        output_image = sys.argv[2] if len(sys.argv) > 2 else "normal_map.png"
        main(input_image, output_image)
