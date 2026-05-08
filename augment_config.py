"""
augment_config.py
-----------------
Albumentations pipeline for building-defect images.

Ultralytics automatically applies this when the file is present in the
project root AND albumentations is installed:
    pip install albumentations

If albumentations is not installed, training still works — Ultralytics
falls back to its built-in augmentations silently.

The pipeline targets:
  • Low-light / over-exposed footage   → CLAHE, RandomBrightness/Contrast
  • Blurry frames                      → MotionBlur, GaussianBlur
  • Outdoor weather variation          → RandomRain, RandomFog, RandomSunFlare
  • Surface texture variation          → RandomGravel, CoarseDropout
"""

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


def get_transform(p: float = 1.0):
    """
    Return an Albumentations Compose transform.

    Args:
        p: Overall pipeline probability. Default 1.0 (always applied).

    Returns:
        A.Compose instance or None if albumentations not installed.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    return A.Compose(
        [
            # ---- Lighting & colour ------------------------------------------
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.3
            ),
            A.ToGray(p=0.05),           # grayscale surveillance footage
            A.ImageCompression(quality_lower=50, quality_upper=100, p=0.2),

            # ---- Blur / noise -----------------------------------------------
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ],
                p=0.35,
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),

            # ---- Weather simulation -----------------------------------------
            A.RandomRain(
                slant_lower=-10, slant_upper=10,
                drop_length=10, drop_width=1, drop_color=(180, 180, 180),
                blur_value=3, brightness_coefficient=0.85,
                rain_type="drizzle", p=0.10,
            ),
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.25, p=0.10),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0, angle_upper=1,
                num_flare_circles_lower=3, num_flare_circles_upper=6,
                src_radius=200, p=0.05,
            ),

            # ---- Surface / texture ------------------------------------------
            # CoarseDropout simulates debris / occlusion patches on walls
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32,
                min_holes=1, min_height=8, min_width=8,
                fill_value=128, p=0.20,
            ),
        ],
        p=p,
        bbox_params=A.BboxParams(
            format="yolo",          # YOLO cx cy w h (normalised)
            label_fields=["class_labels"],
            min_visibility=0.3,     # drop bbox if <30% visible after transform
        ),
    )


# Ultralytics checks for a top-level `transform` symbol in this file
transform = get_transform(p=1.0)
