#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image

OUT_DIR = os.path.join(os.path.dirname(__file__), "textures")
os.makedirs(OUT_DIR, exist_ok=True)

def generate_brick_texture(filename, size=(1024, 1024)):
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    brick_h, brick_w = 64, 160
    mortar_thickness = 8
    
    # Base mortar color (light gray with noise)
    mortar_noise = np.random.normal(180, 15, (size[1], size[0], 3)).clip(150, 210).astype(np.uint8)
    img[:] = mortar_noise

    # Draw brick pattern
    row = 0
    for y in range(0, size[1], brick_h + mortar_thickness):
        offset = (brick_w // 2) if (row % 2 == 1) else 0
        for x in range(-brick_w, size[0] + brick_w, brick_w + mortar_thickness):
            x_start = max(0, x + offset)
            x_end = min(size[0], x + offset + brick_w)
            y_start = max(0, y)
            y_end = min(size[1], y + brick_h)
            if x_start < x_end and y_start < y_end:
                # Brick variation: reddish-brown with surface roughness
                base_r = np.random.randint(140, 180)
                base_g = np.random.randint(60, 90)
                base_b = np.random.randint(40, 70)
                noise = np.random.normal(0, 12, (y_end - y_start, x_end - x_start, 3))
                brick_patch = np.clip(np.array([base_r, base_g, base_b]) + noise, 0, 255).astype(np.uint8)
                img[y_start:y_end, x_start:x_end] = brick_patch
        row += 1
    Image.fromarray(img).save(filename)

def generate_wood_texture(filename, size=(1024, 1024)):
    # Procedural directional wood planks
    y_coords, x_coords = np.indices(size)
    rings = np.sin(x_coords / 15.0 + 3.0 * np.sin(y_coords / 60.0))
    noise = np.random.normal(0, 0.08, size)
    wood_pattern = ((rings + noise + 1.0) * 0.5).clip(0, 1)
    
    base_dark = np.array([120, 75, 40], dtype=np.float32)
    base_light = np.array([190, 135, 85], dtype=np.float32)
    rgb = (wood_pattern[:, :, None] * base_light + (1.0 - wood_pattern[:, :, None]) * base_dark).astype(np.uint8)
    Image.fromarray(rgb).save(filename)

def generate_plaster_texture(filename, size=(512, 512)):
    noise = np.random.normal(235, 10, (size[1], size[0], 3)).clip(210, 255).astype(np.uint8)
    Image.fromarray(noise).save(filename)

def generate_metal_texture(filename, size=(512, 512)):
    base = np.random.normal(160, 8, (size[1], size[0], 3)).clip(130, 190).astype(np.uint8)
    Image.fromarray(base).save(filename)

if __name__ == "__main__":
    generate_brick_texture(os.path.join(OUT_DIR, "brick_wall.png"))
    generate_wood_texture(os.path.join(OUT_DIR, "wood_floor.png"))
    generate_wood_texture(os.path.join(OUT_DIR, "box_wood.png"))
    generate_plaster_texture(os.path.join(OUT_DIR, "plaster_ceiling.png"))
    generate_metal_texture(os.path.join(OUT_DIR, "cylinder_metal.png"))
    print(f"Generated benchmark textures in {OUT_DIR}")
