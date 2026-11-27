"""Real roof detail from the aerial photo: solar panels, skylights, chimneys, dormers, moss and patched tiles.

For every building the photo is first lined up with the footprint (the photo leans and is never exactly on top of
the BAG outline). Then it is tidied up so it reads as a clean rendering rather than a noisy photo: every roof face is
brought to the same brightness and its low-frequency blotches (shadows of trees and neighbours) are flattened, tree
crowns are toned toward the roof color, the blue of sky reflections on panels and roof windows is tamed, and JPEG grain
is removed without blurring edges. The crops are packed into one texture atlas.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from aerial import world_to_px
from config import BUILD, GROUND_RES

SEARCH_M = 2.0  # how far the photo may be off, in meters
TARGET_MAX = 120  # brightest roof the renderer's own sun can take without washing out to pink or white
PEAK_MAX = 140  # and the same for its strongest channel: red tiles clip to pink sooner than grey ones to white
FLAT_SIGMA = 12  # in pixels (3 m): blotches wider than this inside a roof face are shadows, not roof detail


def lum_of(img):
    return img @ np.array([0.299, 0.587, 0.114], np.float32)


def smooth(x, w, sigma):
    """Gaussian blur of x counting only where weight w is."""
    return ndi.gaussian_filter(x * w, sigma) / np.maximum(ndi.gaussian_filter(w, sigma), 1e-3)


def flatten_face(img, lum, m, target):
    """Bring one roof face to the target brightness and remove slow brightness blotches (tree and neighbour shadows),
    leaving anything small: panels, skylights, chimneys, tile texture."""
    gain = np.ones_like(lum)
    med = max(float(np.median(lum[m])), 1.0)
    gain[m] = target / med
    if m.sum() > 40:
        ratio = np.maximum(lum / med, 0.05)
        w = m * np.exp(-((np.log(ratio) / 0.35) ** 2))  # dark panels and bright skylights don't count as "the roof"
        low = smooth(lum, w, FLAT_SIGMA)
        gain[m] = (target / np.maximum(low, 1.0)).clip(0.6, 1.5)[m]
    out = img.copy()
    out[m] = img[m] * gain[m, None]
    return out


def roof_color(img, mask):
    """Typical color of the roof: median of the pixels that are neither trees nor blue glass."""
    px = img[mask]
    ok = (px[:, 1] - np.maximum(px[:, 0], px[:, 2]) < 3) & (px[:, 2] - px[:, 0] < 12)
    return np.median(px[ok] if ok.sum() > 20 else px, axis=0)


def tidy(img, covered, fixed=None):
    """Clean up a roof: tone trees and sky blue down, denoise, keep contrast modest. Monuments get their fixed color."""
    base = roof_color(img, covered)
    lum = lum_of(img)
    med = max(float(lum_of(base)), 1.0)
    # Overhanging tree crowns: greenish pixels on a roof that isn't green itself, pulled to the roof color.
    if base[1] - max(base[0], base[2]) < 6:
        green = img[..., 1] - np.maximum(img[..., 0], img[..., 2])
        a = ndi.gaussian_filter(np.clip((green - 2) / 10, 0, 1) * covered, 0.8).clip(0, 1)[..., None]
        shade = ndi.gaussian_filter(lum / med, 1.5).clip(0.8, 1.05)[..., None]
        img = img * (1 - a) + base * shade * a
        lum = lum_of(img)
    # Sky reflections: panels and roof windows read dark blue-grey, not bright blue.
    blue = np.clip((img[..., 2] - img[..., 0] - 10) / 25, 0, 1)[..., None]
    l = lum[..., None]
    img = l + (img - l) * (1 - 0.75 * blue)  # less saturated
    img = img * (1 - 0.15 * blue - 0.25 * blue * np.clip((l - 0.5 * med) / med, 0, 1))  # and darker
    # Grain: median per channel, then a touch of smoothing.
    img = np.stack([ndi.median_filter(img[..., c], size=3) for c in range(3)], -1)
    img = ndi.gaussian_filter(img, (0.5, 0.5, 0))
    # Modest contrast around the roof color.
    img = base + (img - base) * 0.9
    if base.max() > PEAK_MAX:
        img = img * (PEAK_MAX / float(base.max()))
    if fixed is not None:
        # Monument: tint so the roof as a whole ends up at its known color, detail stays.
        now = np.median(img[covered], axis=0)
        img = img * (np.array(fixed, np.float32) / np.maximum(now, 1.0))
    return np.clip(img, 0, 255)


def fill_outside(img, covered):
    """Spread the nearest roof pixel into the pixels around the roof, so texture filtering at the edge stays roof."""
    idx = ndi.distance_transform_edt(~covered, return_distances=False, return_indices=True)
    return img[idx[0], idx[1]]


PAD = 2  # atlas padding in pixels, so neighbouring crops don't bleed into each other
MAX_TEXTURE = 16384  # the largest texture side of the GPUs the map is rendered on


def gradient(aerial):
    lum = aerial.astype(np.float32).mean(-1)
    return np.hypot(ndi.sobel(lum, 0), ndi.sobel(lum, 1))


def outline_points(ring, step=GROUND_RES):
    pts = []
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        n = max(1, int(np.hypot(*(b - a)) / step))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        pts.append(a + (b - a) * t)
    return np.concatenate(pts)


def best_shift(grad, ring):
    """Shift in meters (RD x, y) that puts the footprint outline on the strongest edges in the photo."""
    px = world_to_px(outline_points(np.asarray(ring, dtype=np.float64)[:, :2]))
    h, w = grad.shape
    steps = np.arange(-SEARCH_M, SEARCH_M + 1e-6, GROUND_RES)
    best, best_score, zero_score = (0.0, 0.0), -1.0, None
    for sx in steps:
        for sy in steps:
            c = np.clip(np.round(px[:, 0] + sx / GROUND_RES).astype(int), 0, w - 1)
            r = np.clip(np.round(px[:, 1] - sy / GROUND_RES).astype(int), 0, h - 1)
            score = float(grad[r, c].mean()) - 0.6 * np.hypot(sx, sy)  # prefer small shifts
            if sx == 0 and sy == 0:
                zero_score = score
            if score > best_score:
                best, best_score = (sx, sy), score
    if zero_score is not None and best_score < zero_score * 1.08:
        return 0.0, 0.0
    return best


def build(aerial, roofs, fixed=None):
    """roofs: list of (key, footprint ring, [roof triangles (M,3,3) in RD per face]); fixed: {key: main roof color}.

    Returns {key: (shift, (x0, y0, w, h) atlas rect, crop origin in mosaic px)} and writes roof_atlas.png."""
    grad = gradient(aerial)
    H, W = aerial.shape[:2]
    crops = []
    for key, ring, faces in roofs:
        if not faces:
            continue
        sx, sy = best_shift(grad, ring)
        tris = np.concatenate(faces)
        pts = world_to_px(tris[..., :2].reshape(-1, 2) + (sx, sy))
        c0, r0 = np.floor(pts.min(0)).astype(int) - 1
        c1, r1 = np.ceil(pts.max(0)).astype(int) + 2
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, W), min(r1, H)
        if c1 - c0 < 2 or r1 - r0 < 2:
            continue
        crop = aerial[r0:r1, c0:c1].astype(np.float32)
        lum = lum_of(crop)
        masks = []
        for f in faces:
            m = Image.new("L", (c1 - c0, r1 - r0), 0)
            d = ImageDraw.Draw(m)
            for t in world_to_px(f[..., :2].reshape(-1, 2) + (sx, sy)).reshape(-1, 3, 2):
                d.polygon([(float(x - c0), float(y - r0)) for x, y in t], fill=1)
            masks.append(np.array(m) > 0)
        covered = np.any(masks, axis=0) if masks else np.zeros(lum.shape, bool)
        if covered.sum() < 4:
            continue
        target = min(float(np.median(lum[covered])), TARGET_MAX)
        out = crop.copy()
        for m in masks:
            if m.sum() >= 4:
                out = flatten_face(out, lum, m, target)
        out = tidy(fill_outside(out, covered), covered, (fixed or {}).get(key))
        crops.append((key, (sx, sy), (c0, r0), out.astype(np.uint8)))

    # Shelf packing, tallest first.
    crops.sort(key=lambda c: -c[3].shape[0])
    # As wide as it takes to stay under MAX_TEXTURE high: a GPU scales a larger texture down, and the roofs blur.
    width = 2048
    while True:
        width *= 2
        x = y = shelf = 0
        placed = []
        for key, shift, origin, img in crops:
            h, w = img.shape[:2]
            if x + w + PAD > width:
                x, y, shelf = 0, y + shelf + PAD, 0
            placed.append((key, shift, origin, img, x, y))
            x += w + PAD
            shelf = max(shelf, h)
        height = int(2 ** np.ceil(np.log2(y + shelf + PAD)))
        if height <= MAX_TEXTURE or width >= MAX_TEXTURE:
            break
    atlas = np.zeros((height, width, 3), np.uint8)
    rects = {}
    for key, shift, origin, img, x, y in placed:
        h, w = img.shape[:2]
        atlas[y : y + h, x : x + w] = img
        # Repeat the edge pixels into the padding.
        atlas[y : y + h, x + w : x + w + PAD] = img[:, -1:]
        atlas[y + h : y + h + PAD, x : x + w] = img[-1:, :]
        rects[key] = (shift, (x, y), origin)
    Image.fromarray(atlas).save(BUILD / "roof_atlas.png")
    return rects, (width, height)


def atlas_uv(rd_xy, rect, size):
    """Atlas texture coordinates (u right, v up as in WebGL with flipY) for RD points on a roof."""
    (sx, sy), (ax, ay), (c0, r0) = rect
    px = world_to_px(np.asarray(rd_xy) + (sx, sy))
    u = (ax + px[:, 0] - c0) / size[0]
    v = 1.0 - (ay + px[:, 1] - r0) / size[1]
    return u, v
