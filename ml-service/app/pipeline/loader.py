"""
loader.py — format-aware image loader, upstream of preprocess.py.

Dispatches by file extension:
  .png / .jpg / .jpeg / .bmp  -> cv2.imread (byte-identical to the old path)
  .tif / .tiff                -> rasterio (GeoTIFF-aware)
  .xml                        -> PDS4 detached label, opened via rasterio
  .lbl                        -> PDS3 detached label, opened via rasterio
  .img                        -> redirected to a sibling .xml (PDS4) or .lbl
                                  (PDS3) label in the same directory

Every raised error is a ValueError so callers that already catch ValueError
around the old cv2.imread path (see app/main.py) keep working unchanged.
"""
import logging
import os
import re

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CV2_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
_RASTER_EXTENSIONS = {".tif", ".tiff", ".xml", ".lbl"}

# ch2_<inst>_<phase><datatype><camera>_<timestamp>_...
# e.g. ch2_tmc_nca_20200207T0716469418_d_img_d18.xml
_CH2_FILENAME_RE = re.compile(
    r"^ch2_(?P<instrument>[a-z0-9]+)_(?P<phase>[a-z])(?P<datatype>[a-z])(?P<camera>[a-z])_"
    r"(?P<timestamp>\d{8}T\d+)_",
    re.IGNORECASE,
)
_DATATYPE_MAP = {"r": "raw", "c": "calibrated", "d": "derived"}
_CAMERA_MAP = {"f": "fore", "n": "nadir", "a": "aft", "p": "panchromatic"}


def _parse_product_id(path: str) -> dict | None:
    """Parse the ch2_<inst>_<phase><datatype><camera>_<timestamp>_... filename
    convention. Returns None for non-conforming names rather than raising."""
    name = os.path.basename(path)
    m = _CH2_FILENAME_RE.match(name)
    if not m:
        return None
    g = m.groupdict()
    datatype_letter = g["datatype"].lower()
    camera_letter = g["camera"].lower()
    return {
        "instrument": g["instrument"].lower(),
        "phase_code": g["phase"].lower(),
        "calibration": _DATATYPE_MAP.get(datatype_letter, datatype_letter),
        "camera": _CAMERA_MAP.get(camera_letter, camera_letter),
        "timestamp": g["timestamp"],
        "filename": name,
    }


def _stretch_to_uint8(arr: np.ndarray, bounds: tuple[float, float] | None):
    """Convert a non-8-bit array to uint8. Returns (uint8_array, method, bounds_used).

    A 2-98 percentile linear stretch is used instead of a naive bit-shift or
    full-range min/max: lunar scenes have shadowed crater floors and sunlit
    rims at opposite histogram extremes, and a full-range map wastes most of
    the 8-bit space on values that never occur in the scene.
    """
    if arr.dtype == np.uint8:
        return arr, "none", None

    if bounds is not None:
        lo, hi = float(bounds[0]), float(bounds[1])
        method = "percentile_2_98"
    else:
        lo, hi = (float(v) for v in np.percentile(arr, [2, 98]))
        method = "percentile_2_98"
        if hi <= lo:
            # Degenerate histogram (e.g. a near-constant test fixture) — fall
            # back to the actual min/max rather than dividing by zero.
            lo, hi = float(arr.min()), float(arr.max())
            method = "minmax"
            if hi <= lo:
                hi = lo + 1.0

    clipped = np.clip(arr.astype(np.float64), lo, hi)
    scaled = ((clipped - lo) / (hi - lo) * 255.0).round().astype(np.uint8)
    return scaled, method, (lo, hi)


def _load_via_cv2(path: str, ext: str, product_id: dict | None):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image from path: {path}")
    fmt = "jpeg" if ext in (".jpg", ".jpeg") else ext.lstrip(".")
    metadata = {
        "source_format": fmt,
        "original_dtype": str(img.dtype),
        "band_count": 1,
        "band_used": 1,
        "subdataset_used": None,
        "original_shape": img.shape,
        "crs": None,
        "transform": None,
        "scaling_applied": "none",
        "stretch_bounds": None,
        "product_id": product_id,
    }
    return img, metadata


def _open_raster_dataset(path: str, rasterio_mod):
    """Open path with rasterio, following into the first subdataset if the
    top-level dataset exposes zero bands (a label with multiple Array objects)."""
    ds = rasterio_mod.open(path)
    if ds.count == 0:
        subs = list(ds.subdatasets)
        ds.close()
        if not subs:
            raise ValueError(f"'{path}' has no raster bands and no subdatasets.")
        chosen = subs[0]
        logger.warning(
            "%s exposes %d subdataset(s) with no top-level band; using subdataset %r",
            path, len(subs), chosen,
        )
        ds = rasterio_mod.open(chosen)
        return ds, chosen
    return ds, None


def _load_via_rasterio(
    path: str, ext: str, band: int | None, stretch_bounds: tuple[float, float] | None,
    product_id: dict | None,
):
    try:
        import rasterio
    except ImportError as e:
        raise ValueError(
            f"rasterio is required to read '{path}' but is not installed: {e}"
        ) from e

    try:
        ds, subdataset_used = _open_raster_dataset(path, rasterio)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not open raster '{path}': {e}") from e

    try:
        band_count = ds.count
        band_used = band if band is not None else 1
        if band_count > 1 and band is None:
            logger.warning(
                "%s has %d bands; no band specified, defaulting to band 1", path, band_count
            )
        if not (1 <= band_used <= band_count):
            raise ValueError(
                f"Requested band {band_used} out of range for '{path}' with {band_count} band(s)."
            )
        try:
            arr = ds.read(band_used)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Could not read band {band_used} from '{path}': {e}") from e

        original_dtype = str(arr.dtype)
        original_shape = arr.shape
        crs = str(ds.crs) if ds.crs else None
        transform = tuple(ds.transform)[:6] if ds.transform else None
    finally:
        ds.close()

    img_u8, scaling_applied, bounds_used = _stretch_to_uint8(arr, stretch_bounds)

    if ext in (".tif", ".tiff"):
        source_format = "geotiff" if crs else "tiff"
    elif ext == ".xml":
        source_format = "pds4"
    elif ext == ".lbl":
        source_format = "pds3"
    else:
        source_format = ext.lstrip(".")

    metadata = {
        "source_format": source_format,
        "original_dtype": original_dtype,
        "band_count": band_count,
        "band_used": band_used,
        "subdataset_used": subdataset_used,
        "original_shape": original_shape,
        "crs": crs,
        "transform": transform,
        "scaling_applied": scaling_applied,
        "stretch_bounds": bounds_used,
        "product_id": product_id,
    }
    return img_u8, metadata


def _load_img_redirect(path: str, band: int | None, stretch_bounds: tuple[float, float] | None):
    """A raw .img carries no self-describing header — the label next to it
    (PDS4 .xml, or PDS3 .lbl) says how to interpret the bytes. Redirect to
    whichever sibling label exists; the user will naturally point at the
    .img, so this must not require them to know about the label file."""
    stem, _ = os.path.splitext(path)
    xml_candidates = [stem + ".xml", stem + ".XML"]
    for candidate in xml_candidates:
        if os.path.isfile(candidate):
            logger.info("Redirecting '%s' -> detached PDS4 label '%s'", path, candidate)
            return load_image(candidate, band=band, stretch_bounds=stretch_bounds)

    lbl_candidates = [stem + ".lbl", stem + ".LBL"]
    for candidate in lbl_candidates:
        if os.path.isfile(candidate):
            logger.info("Redirecting '%s' -> detached PDS3 label '%s'", path, candidate)
            return load_image(candidate, band=band, stretch_bounds=stretch_bounds)

    raise ValueError(
        f"'{path}' is a raw .img file with no detached label next to it. Looked for: "
        f"{', '.join(xml_candidates + lbl_candidates)}. PDS4 products need the sibling "
        f".xml label; PDS3 (Chandrayaan-1-era) products need the sibling .lbl label."
    )


def load_image(
    path: str,
    band: int | None = None,
    stretch_bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Returns (uint8 single-channel 2D array, metadata dict).

    metadata keys (always present, None where unknown):
      source_format    : "png" | "jpeg" | "bmp" | "tiff" | "geotiff" | "pds4" | "pds3"
      original_dtype    : e.g. "uint16"
      band_count        : int
      band_used         : int (1-indexed)
      subdataset_used   : str | None
      original_shape    : (h, w)
      crs               : CRS string or None
      transform         : 6-tuple affine or None
      scaling_applied   : "none" | "percentile_2_98" | "minmax"
      stretch_bounds    : (lo, hi) actually used, or None
      product_id        : dict parsed from a ch2_* filename, else None
    """
    ext = os.path.splitext(path)[1].lower()
    product_id = _parse_product_id(path)

    if ext == ".img":
        return _load_img_redirect(path, band, stretch_bounds)

    if ext in _CV2_EXTENSIONS:
        return _load_via_cv2(path, ext, product_id)

    if ext in _RASTER_EXTENSIONS:
        return _load_via_rasterio(path, ext, band, stretch_bounds, product_id)

    raise ValueError(
        f"Unsupported image format '{ext}' for '{path}'. Supported: "
        ".png, .jpg, .jpeg, .bmp, .tif, .tiff, .xml (PDS4 label), "
        ".img (PDS4/PDS3, redirected to its sibling label), .lbl (PDS3 label)."
    )
