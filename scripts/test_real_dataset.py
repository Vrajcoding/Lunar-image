"""
test_real_dataset.py — Real Lunar Dataset Testing & Evaluation CLI Tool

Enables testing LunarMatch AI on user-supplied real lunar image pairs (OHRC, TMC-2, IIRS, LROC, GeoTIFF, PNG, JPG).

Usage:
  # 1. Test a single real pair:
  python scripts/test_real_dataset.py --source path/to/source.png --reference path/to/reference.png

  # 2. Test with Chandrayaan-2 PDS4 XML metadata:
  python scripts/test_real_dataset.py --source source.tif --reference ref.tif --source-xml source.xml --reference-xml ref.xml

  # 3. Test an entire directory of pairs:
  python scripts/test_real_dataset.py --pairs-dir datasets/chandrayaan2/pairs/
"""
import argparse
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

# Add ml-service to path
ml_service_dir = Path(__file__).resolve().parent.parent / "ml-service"
sys.path.insert(0, str(ml_service_dir))

from app.pipeline.registration import run_registration_pipeline
from app.pipeline.loader import load_image, _extract_xml_metadata


def evaluate_single_pair(
    source_path: str,
    reference_path: str,
    source_xml: str | None = None,
    reference_xml: str | None = None,
    output_dir: str = "experiments/results/real_tests",
    detector: str = "auto",
    use_subpixel: bool = True,
):
    """Run full registration pipeline on a single real image pair."""
    os.makedirs(output_dir, exist_ok=True)
    pair_name = f"{Path(source_path).stem}_vs_{Path(reference_path).stem}"
    pair_output_dir = os.path.join(output_dir, pair_name)
    os.makedirs(pair_output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Testing Real Pair: {Path(source_path).name} vs {Path(reference_path).name}")
    print(f"=======================================================")

    # Load source and reference images
    src_img, src_meta = load_image(source_path)
    ref_img, ref_meta = load_image(reference_path)

    if source_xml and os.path.exists(source_xml):
        src_meta.update(_extract_xml_metadata(source_xml))
    if reference_xml and os.path.exists(reference_xml):
        ref_meta.update(_extract_xml_metadata(reference_xml))

    print(f"Source Image   : {src_img.shape[1]}x{src_img.shape[0]} px | Payload: {src_meta.get('instrument', 'Unknown')}")
    print(f"Reference Image: {ref_img.shape[1]}x{ref_img.shape[0]} px | Payload: {ref_meta.get('instrument', 'Unknown')}")

    if src_meta.get("solar_zenith_deg") is not None:
        print(f"Source Solar Zenith: {src_meta['solar_zenith_deg']:.1f}° | Ref Solar Zenith: {ref_meta.get('solar_zenith_deg', 'N/A')}°")

    # Run registration
    result = run_registration_pipeline(
        src_path=source_path,
        ref_path=reference_path,
        job_dir=pair_output_dir,
        mode=detector,
        src_img_u8=src_img,
        src_meta=src_meta,
        ref_img_u8=ref_img,
        ref_meta=ref_meta,
    )

    # Print Results
    status = result.get("status", "unknown")
    metrics = result.get("metrics", {})
    method_used = result.get("method", "N/A")
    conf_level = metrics.get("confidence_level", "N/A")
    conf_score = metrics.get("confidence_score", 0.0)

    print("\n--- Registration Results ---")
    print(f"Status           : {status.upper()}")
    print(f"Method Used      : {method_used}")
    print(f"Confidence       : {conf_level} ({conf_score:.3f})")
    
    print(f"Total Matches    : {metrics.get('total_matches', 0)}")
    print(f"Inlier Count     : {metrics.get('inlier_count', 0)}")
    inlier_ratio = metrics.get('inlier_ratio')
    if inlier_ratio is not None:
        print(f"Inlier Ratio     : {inlier_ratio * 100:.1f}%")
    else:
        print(f"Inlier Ratio     : N/A")
    print(f"Fit RMSE         : {metrics.get('rmse', metrics.get('fit_rmse_px', 'N/A'))} px")
    print(f"Independent Test : {metrics.get('test_rmse_px', 'N/A')} px")
    print(f"Spatial Entropy  : {metrics.get('spatial_entropy', 'N/A')}")
    print(f"Runtime          : {metrics.get('runtime_sec', 0.0):.2f} seconds")

    print("\n--- Generated Output Artifacts ---")
    print(f"  - Registered Image   : {result.get('registered_image_path')}")
    print(f"  - Difference Heatmap : {result.get('difference_image_path')}")
    print(f"  - Match Points CSV   : {result.get('match_points_path')}")

    # Save summary json
    summary_path = os.path.join(pair_output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  - Full Report JSON   : {summary_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test LunarMatch AI on real lunar image datasets.")
    parser.add_argument("--source", type=str, help="Path to real source image (PNG, TIFF, JPG)")
    parser.add_argument("--reference", type=str, help="Path to real reference image (PNG, TIFF, JPG)")
    parser.add_argument("--source-xml", type=str, default=None, help="Optional path to PDS4 XML label for source")
    parser.add_argument("--reference-xml", type=str, default=None, help="Optional path to PDS4 XML label for reference")
    parser.add_argument("--pairs-dir", type=str, default=None, help="Directory containing subdirectories or pairs of images")
    parser.add_argument("--detector", type=str, default="hybrid", choices=["hybrid", "loftr", "sift", "orb"])
    parser.add_argument("--no-subpixel", action="store_true", help="Disable subpixel refinement")
    parser.add_argument("--output-dir", type=str, default="experiments/results/real_tests")

    args = parser.parse_args()

    if args.pairs_dir:
        pairs_dir = Path(args.pairs_dir)
        subdirs = [d for d in pairs_dir.iterdir() if d.is_dir()]
        if subdirs:
            for sd in subdirs:
                src = list(sd.glob("*src*")) or list(sd.glob("*source*"))
                ref = list(sd.glob("*ref*")) or list(sd.glob("*reference*"))
                if src and ref:
                    src_xml = list(sd.glob("*src*.xml")) or list(sd.glob("*source*.xml"))
                    ref_xml = list(sd.glob("*ref*.xml")) or list(sd.glob("*reference*.xml"))
                    evaluate_single_pair(
                        str(src[0]), str(ref[0]),
                        source_xml=str(src_xml[0]) if src_xml else None,
                        reference_xml=str(ref_xml[0]) if ref_xml else None,
                        output_dir=args.output_dir,
                        detector=args.detector,
                        use_subpixel=not args.no_subpixel,
                    )
        else:
            print("Please specify --source and --reference or organize subfolders with *src* and *ref* images.")
    elif args.source and args.reference:
        evaluate_single_pair(
            args.source,
            args.reference,
            args.source_xml,
            args.reference_xml,
            output_dir=args.output_dir,
            detector=args.detector,
            use_subpixel=not args.no_subpixel,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
