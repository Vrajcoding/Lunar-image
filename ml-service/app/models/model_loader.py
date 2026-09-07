"""
model_loader.py — Model lifecycle and singleton cache manager for learned neural matchers.
Ensures neural weights are loaded once in memory and reused across concurrent requests.
"""
import logging
from typing import Optional
from app.models.loftr_matcher import LoFTRMatcher
from app.config import DEVICE, LOFTR_ENABLED

logger = logging.getLogger(__name__)

_LOFTR_INSTANCE: Optional[LoFTRMatcher] = None


def get_loftr_matcher(device: Optional[str] = None, force_reload: bool = False) -> Optional[LoFTRMatcher]:
    """Retrieve the singleton LoFTRMatcher instance.
    
    If LOFTR_ENABLED is False, returns None.
    If already initialized, returns cached instance.
    """
    global _LOFTR_INSTANCE

    if not LOFTR_ENABLED:
        logger.info("LoFTR is disabled by configuration (LOFTR_ENABLED=False)")
        return None

    if _LOFTR_INSTANCE is not None and not force_reload:
        return _LOFTR_INSTANCE

    chosen_device = device or DEVICE
    logger.info("Initializing singleton LoFTRMatcher on device '%s'...", chosen_device)
    
    try:
        matcher = LoFTRMatcher(device=chosen_device)
        if matcher.is_available:
            _LOFTR_INSTANCE = matcher
            return _LOFTR_INSTANCE
        else:
            logger.warning("LoFTRMatcher model failed to load, returning None.")
            return None
    except Exception as e:
        logger.error("Error creating LoFTRMatcher instance: %s", e)
        return None
