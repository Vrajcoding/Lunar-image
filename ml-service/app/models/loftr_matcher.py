"""
loftr_matcher.py — Deep learning correspondence wrapper for LoFTR (Detector-Free Local Feature Matching).
Uses Kornia's pretrained LoFTR model with support for CPU/CUDA and sub-pixel accuracy.
"""
import logging
import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class LoFTRMatcher:
    """LoFTR (Local Feature TRansformer) wrapper for lunar image correspondence."""

    def __init__(self, device: str | torch.device | None = None, pretrained: str = "outdoor"):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            if device.lower() == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        self.pretrained = pretrained
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            import ssl
            # Bypass strict Windows cert validation when fetching weights from torch hub
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
            except Exception:
                pass

            import kornia.feature as kf
            logger.info("Initializing LoFTR model (pretrained=%s, device=%s)...", self.pretrained, self.device)
            self._model = kf.LoFTR(pretrained=self.pretrained).to(self.device)
            self._model.eval()
            logger.info("LoFTR model successfully loaded on %s", self.device)
        except Exception as e:
            logger.error("Failed to load LoFTR model: %s", e)
            self._model = None

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def _preprocess_tensor(self, img: np.ndarray) -> tuple[torch.Tensor, tuple[int, int], tuple[float, float]]:
        """Convert a 2D uint8/float grayscale image to a torch tensor divisible by 8.
        Returns (tensor, original_shape, (scale_x, scale_y))."""
        if img.ndim == 3:
            if img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = img[:, :, 0]
        else:
            gray = img

        h, w = gray.shape[:2]
        
        # LoFTR requires dimensions to be divisible by 8.
        # Resize to nearest multiples of 8 (or max dimension if downscaling large orbital scenes)
        MAX_DIM = 1200
        scale = 1.0
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
        
        target_w = int(round(w * scale / 8.0) * 8)
        target_h = int(round(h * scale / 8.0) * 8)
        target_w = max(target_w, 8)
        target_h = max(target_h, 8)

        if target_w != w or target_h != h:
            resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        else:
            resized = gray

        # Normalize to [0, 1] float32 tensor of shape (1, 1, H, W)
        if resized.dtype == np.uint8:
            arr = resized.astype(np.float32) / 255.0
        else:
            arr = resized.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / (arr.max() + 1e-6)

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)
        scale_x = w / float(target_w)
        scale_y = h / float(target_h)

        return tensor, (h, w), (scale_x, scale_y)

    def match(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        confidence_threshold: float = 0.2,
    ) -> dict:
        """Run LoFTR correspondence between source (image0) and reference (image1).
        
        Returns:
            dict containing:
                "points0": (N, 2) np.ndarray of float coordinates in image0
                "points1": (N, 2) np.ndarray of float coordinates in image1
                "confidence": (N,) np.ndarray of confidence scores in [0, 1]
                "method": "loftr"
        """
        if not self.is_available:
            logger.warning("LoFTR model not available, returning empty matches.")
            return {
                "points0": np.zeros((0, 2), dtype=np.float32),
                "points1": np.zeros((0, 2), dtype=np.float32),
                "confidence": np.zeros((0,), dtype=np.float32),
                "method": "loftr",
            }

        try:
            t0, orig_shape0, (scale_x0, scale_y0) = self._preprocess_tensor(image0)
            t1, orig_shape1, (scale_x1, scale_y1) = self._preprocess_tensor(image1)

            input_dict = {"image0": t0, "image1": t1}

            with torch.no_grad():
                correspondences = self._model(input_dict)

            kpts0 = correspondences.get("keypoints0", None)
            kpts1 = correspondences.get("keypoints1", None)
            conf = correspondences.get("confidence", None)

            if kpts0 is None or kpts1 is None or len(kpts0) == 0:
                return {
                    "points0": np.zeros((0, 2), dtype=np.float32),
                    "points1": np.zeros((0, 2), dtype=np.float32),
                    "confidence": np.zeros((0,), dtype=np.float32),
                    "method": "loftr",
                }

            pts0_np = kpts0.detach().cpu().numpy().astype(np.float32)
            pts1_np = kpts1.detach().cpu().numpy().astype(np.float32)
            conf_np = conf.detach().cpu().numpy().astype(np.float32) if conf is not None else np.ones(len(pts0_np), dtype=np.float32)

            # Scale keypoints back to original input coordinates
            pts0_np[:, 0] *= scale_x0
            pts0_np[:, 1] *= scale_y0
            pts1_np[:, 0] *= scale_x1
            pts1_np[:, 1] *= scale_y1

            # Filter by confidence threshold
            if confidence_threshold > 0.0:
                mask = conf_np >= confidence_threshold
                pts0_np = pts0_np[mask]
                pts1_np = pts1_np[mask]
                conf_np = conf_np[mask]

            return {
                "points0": pts0_np,
                "points1": pts1_np,
                "confidence": conf_np,
                "method": "loftr",
            }

        except Exception as e:
            logger.error("Error during LoFTR matching: %s", e)
            return {
                "points0": np.zeros((0, 2), dtype=np.float32),
                "points1": np.zeros((0, 2), dtype=np.float32),
                "confidence": np.zeros((0,), dtype=np.float32),
                "method": "loftr",
            }
