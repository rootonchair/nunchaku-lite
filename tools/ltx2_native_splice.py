"""Write a native-layout Nunchaku Lite LTX-2.x transformer for ltx-core / ComfyUI.

Splices the SVDQ layers of a Nunchaku Lite checkpoint (e.g. a Diffusers-layout export) into Lightricks' official
bf16 transformer file, then verifies the result (non-target tensors and metadata byte-identical to the official
file, SVDQ tensors byte-identical to the source)::

    python tools/ltx2_native_splice.py \\
        --official ltx-2.5-22b-distilled-transformer-bf16.safetensors \\
        --quantized nvfp4-r32/transformer/diffusion_pytorch_model.safetensors \\
        --out ltx-2.5-22b-distilled-transformer-nunchaku-nvfp4-r32.safetensors

``--targets-regex`` splices only the quantized layers whose name fully matches it; the rest stay bf16.
"""

from __future__ import annotations

import argparse
import json
import re

from nunchaku_lite.loader import read_checkpoint_spec
from nunchaku_lite.native_export import splice_native, verify_native_splice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--official", required=True, help="Lightricks' bf16 transformer .safetensors")
    parser.add_argument("--quantized", required=True, help="Nunchaku Lite checkpoint holding the SVDQ layers")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-fp4-rotation", action="store_true")
    parser.add_argument("--targets-regex", help="splice only quantized layers whose full name matches")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    targets = None
    if args.targets_regex is not None:
        pattern = re.compile(args.targets_regex)
        spec = read_checkpoint_spec(args.quantized, allow_fp4_rotation=args.allow_fp4_rotation)
        targets = sorted(name for name in spec.layers if pattern.fullmatch(name))
    report = splice_native(
        args.official, args.quantized, args.out, allow_fp4_rotation=args.allow_fp4_rotation, targets=targets
    )
    if not args.skip_verify:
        verify_native_splice(args.official, args.out, args.quantized, targets=targets)
        report["verified"] = True
    print(json.dumps(report))


if __name__ == "__main__":
    main()
