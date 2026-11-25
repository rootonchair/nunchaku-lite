"""Write a native-layout Nunchaku Lite LTX-2.x transformer for ltx-core / ComfyUI.

Splices the SVDQ layers of a Nunchaku Lite checkpoint (e.g. a Diffusers-layout export) into Lightricks' official
bf16 transformer file, then verifies the result (non-target tensors and metadata byte-identical to the official
file, SVDQ tensors byte-identical to the source)::

    python tools/ltx2_native_splice.py \\
        --official ltx-2.5-22b-distilled-transformer-bf16.safetensors \\
        --quantized nvfp4-r32/transformer/diffusion_pytorch_model.safetensors \\
        --out ltx-2.5-22b-distilled-transformer-nunchaku-nvfp4-r32.safetensors
"""

from __future__ import annotations

import argparse
import json

from nunchaku_lite.native_export import splice_native, verify_native_splice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--official", required=True, help="Lightricks' bf16 transformer .safetensors")
    parser.add_argument("--quantized", required=True, help="Nunchaku Lite checkpoint holding the SVDQ layers")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-fp4-rotation", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    report = splice_native(args.official, args.quantized, args.out, allow_fp4_rotation=args.allow_fp4_rotation)
    if not args.skip_verify:
        verify_native_splice(args.official, args.out, args.quantized)
        report["verified"] = True
    print(json.dumps(report))


if __name__ == "__main__":
    main()
