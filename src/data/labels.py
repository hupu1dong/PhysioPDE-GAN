"""PTB-XL label mapping to the 7 paper abnormality classes (reproduction mapping)."""
from __future__ import annotations

import ast
from typing import Dict, List

# The 7 classes of Table 1 (G12EC) and Table 2 (PTB-XL)
# Each class maps to a set of PTB-XL SCP-ECG codes (any hit is positive)
CLASSES: List[str] = ["IAVB", "RBBB", "LBBB", "NSIVCB", "LAnFB", "QAb", "AFL"]
NORM = "NORM"
ALL_LABELS: List[str] = CLASSES + [NORM]

PAPER_TO_SCP: Dict[str, List[str]] = {
    "IAVB": ["1AVB"],                 # first degree AV block
    "RBBB": ["CRBBB", "IRBBB"],        # Table 1 uses RBBB; PTB-XL splits complete/incomplete
    "LBBB": ["CLBBB", "ILBBB"],
    "NSIVCB": ["IVCD"],                # non-specific intraventricular conduction disturbance
    "LAnFB": ["LAFB"],                 # left anterior fascicular block
    "QAb": ["QWAVE"],                  # paper QAb = pathological Q; PTB-XL has no QAB code, QWAVE is used
    "AFL": ["AFLT"],                   # atrial flutter
}


def parse_scp_codes(raw: str) -> Dict[str, float]:
    """The scp_codes column of ptbxl_database.csv is a stringified dict."""
    try:
        d = ast.literal_eval(raw)
        return {str(k): float(v) for k, v in d.items()}
    except (ValueError, SyntaxError):
        return {}


def codes_to_multihot(codes: Dict[str, float], classes: List[str] = CLASSES) -> List[int]:
    return [int(any(c in codes for c in PAPER_TO_SCP[c])) for c in classes]


def is_norm(codes: Dict[str, float]) -> int:
    return int("NORM" in codes)
