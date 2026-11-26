"""Convert the LTX-2.5 distilled model from Lightricks/LTX-2.5 into the Diffusers folder layout.

The nunchaku-lite LTX-2.5 example (docs/models/ltx2.md) loads every component from such a folder. Lightricks ships
LTX-2.5 as single-file checkpoints; this script reuses the per-component converters of Diffusers'
``scripts/convert_ltx2_to_diffusers.py`` (version "2.5") and adds what that repo's file split needs:

- transformer and connectors come from the distilled transformer file; the connectors' text projections live in the
  text-encoder file and are merged in;
- the text encoder (Lightricks' fine-tuned Gemma 4 12B) is re-keyed to the Hugging Face Gemma 4 layout, with the
  config from google/gemma-4-12B and the tokenizer embedded in Lightricks' file;
- vae, diffusion decoder, audio VAE, vocoder, duration head and the distilled scheduler.

Usage::

    hf auth login  # Lightricks/LTX-2.5 is gated: accept its license on the Hub first
    python tools/convert_ltx25_distilled.py --out ltx-2.5-distilled-diffusers

Downloads 72 GB from Lightricks/LTX-2.5 (or reads the files from ``--src``) and writes about 70 GB. The transformer
step holds the 42 GB source file and its converted copy, so plan for about 85 GB of CPU memory. ``--meta`` checks
every key and shape from the safetensors headers only, without downloading or writing weights. Tested with Diffusers
0.43.0.

The converted model is a Derivative of LTX-2.5 and stays under the LTX-2.x Community License.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import tempfile
import urllib.request

import torch

REPO = "Lightricks/LTX-2.5"
GEMMA = "google/gemma-4-12B"
FILES = {
    "dit": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "te": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "dvae": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "duration": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
}
STEPS = ("vae", "diffusion_decoder", "audio", "duration_head", "text_encoder", "connectors", "transformer", "scheduler")
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "U8": torch.uint8, "I64": torch.int64}
CONVERTER_URL = "https://raw.githubusercontent.com/huggingface/diffusers/{ref}/scripts/convert_ltx2_to_diffusers.py"


def te_key_to_hf(key: str) -> str | None:
    """Lightricks text-encoder key -> Hugging Face Gemma 4 key, or None for entries that are not encoder weights."""
    for old, new in (
        ("model.", "model.language_model."),
        ("vision_model.", "model.vision_embedder."),
        ("audio_projector.", "model.embed_audio."),
        ("multi_modal_projector.", "model.embed_vision."),
    ):
        if key.startswith(old):
            return new + key.removeprefix(old)
    return None  # text_embedding_projection.* (-> connectors), embedded tokenizer/config assets


def _header_from_bytes(read) -> dict:
    n = struct.unpack("<Q", read(0, 8))[0]
    header = json.loads(read(8, 8 + n))
    header.pop("__metadata__", None)
    return header


def local_header(path: str) -> dict:
    def read(a, b):
        with open(path, "rb") as f:
            f.seek(a)
            return f.read(b - a)

    return _header_from_bytes(read)


def hub_header(repo: str, filename: str) -> dict:
    from huggingface_hub import get_session, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    def read(a, b):
        r = get_session().get(
            hf_hub_url(repo, filename), headers={**build_hf_headers(), "Range": f"bytes={a}-{b - 1}"}, timeout=120
        )
        r.raise_for_status()
        return r.content

    return _header_from_bytes(read)


def load_converter(path: str | None):
    """Diffusers' LTX-2 conversion script: ``path``; else the one in the source checkout of an editable Diffusers
    install; else the copy from the installed Diffusers release tag (or main for a dev build)."""
    if path is None:
        import diffusers

        local = os.path.join(os.path.dirname(diffusers.__file__), "..", "..", "scripts", "convert_ltx2_to_diffusers.py")
        if os.path.exists(local):
            path = os.path.normpath(local)
        else:
            ref = f"v{diffusers.__version__}" if ".dev" not in diffusers.__version__ else "main"
            path = os.path.join(tempfile.mkdtemp(), "convert_ltx2_to_diffusers.py")
            urllib.request.urlretrieve(CONVERTER_URL.format(ref=ref), path)
        print(f"using {path if os.path.exists(local) else 'Diffusers ' + ref + ' convert_ltx2_to_diffusers.py'}")
    spec = importlib.util.spec_from_file_location("convert_ltx2_to_diffusers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="output folder (Diffusers layout)")
    p.add_argument(
        "--src", help="local folder with the Lightricks/LTX-2.5 files at their repo paths (default: download)"
    )
    p.add_argument("--converter", help="path to Diffusers' scripts/convert_ltx2_to_diffusers.py (default: download)")
    p.add_argument("--meta", action="store_true", help="check keys and shapes from file headers only")
    p.add_argument("--only", default=",".join(STEPS), help=f"comma-separated subset of: {','.join(STEPS)}")
    args = p.parse_args()
    only = set(args.only.split(","))
    if only - set(STEPS):
        p.error(f"unknown steps {sorted(only - set(STEPS))}")

    conv = load_converter(args.converter)
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig

    def load(name: str) -> dict:
        rel = FILES[name]
        if args.meta:
            header = local_header(os.path.join(args.src, rel)) if args.src else hub_header(REPO, rel)
            return {k: torch.empty(v["shape"], dtype=DTYPES[v["dtype"]], device="meta") for k, v in header.items()}
        if args.src:
            return load_file(os.path.join(args.src, rel))
        from huggingface_hub import hf_hub_download

        return load_file(hf_hub_download(REPO, rel))

    def save(model, sub: str) -> None:
        if args.meta:
            n = sum(t.numel() for t in model.state_dict().values())
            print(f"[meta] {sub}: OK ({n / 1e9:.3f}B params)", flush=True)
            return
        model.to(torch.bfloat16).save_pretrained(os.path.join(args.out, sub), max_shard_size="5GB")
        print(f"saved {sub}", flush=True)

    prefix = conv.get_model_state_dict_from_combined_ckpt
    gemma_cfg = AutoConfig.from_pretrained(GEMMA)

    if "vae" in only:
        save(conv.convert_ltx2_video_vae(load("vae"), version="2.5", timestep_conditioning=False), "vae")
    if "diffusion_decoder" in only:
        state = load("dvae")
        # only the keyframe-stream decode path reads `decoder.type_emb`; plain decoding never uses it, and the
        # Diffusers model has no such parameter
        state.pop("decoder.type_emb", None)
        save(conv.convert_ltx2_diffusion_video_vae(state, version="2.5"), "diffusion_decoder")
    if "audio" in only:
        state = load("audio")
        save(conv.convert_ltx2_audio_vae(prefix(state, "audio_vae."), version="2.5"), "audio_vae")
        save(conv.convert_ltx2_vocoder(prefix(state, "vocoder."), version="2.5"), "vocoder")
        del state
    if "duration_head" in only:
        head = conv.convert_ltx2_duration_head(prefix(load("duration"), "duration_head."))
        if head is None:
            raise SystemExit("duration head missing from the model_patches file")
        save(head, "duration_head")

    text_projections = {}
    if only & {"text_encoder", "connectors"}:
        te = load("te")
        text_projections = {k: v for k, v in te.items() if k.startswith("text_embedding_projection.")}
        if len(text_projections) != 4:
            raise SystemExit(f"expected 4 text projection tensors, found {sorted(text_projections)}")
        if "text_encoder" in only:
            hf_state = {te_key_to_hf(k): v for k, v in te.items() if te_key_to_hf(k) is not None}
            # must match google/gemma-4-12B's on-disk keys, shapes and dtypes exactly
            ref = hub_header(GEMMA, "model.safetensors")
            missing, unexpected = set(ref) - set(hf_state), set(hf_state) - set(ref)
            mismatched = [
                k
                for k in set(hf_state) & set(ref)
                if list(hf_state[k].shape) != ref[k]["shape"] or hf_state[k].dtype != DTYPES[ref[k]["dtype"]]
            ]
            if missing or unexpected or mismatched:
                raise SystemExit(
                    f"text encoder does not match {GEMMA}: missing {sorted(missing)[:5]}, "
                    f"unexpected {sorted(unexpected)[:5]}, mismatched {sorted(mismatched)[:5]}"
                )
            print(f"[{'meta' if args.meta else 'ok'}] text_encoder: {len(hf_state)} tensors match {GEMMA}", flush=True)
            if not args.meta:
                te_dir, tok_dir = os.path.join(args.out, "text_encoder"), os.path.join(args.out, "tokenizer")
                os.makedirs(te_dir, exist_ok=True)
                os.makedirs(tok_dir, exist_ok=True)
                save_file(
                    {k: v.contiguous() for k, v in hf_state.items()},
                    os.path.join(te_dir, "model.safetensors"),
                    metadata={"format": "pt"},
                )
                gemma_cfg.save_pretrained(te_dir)

                # Lightricks embeds its tokenizer and generation config in the encoder file as uint8 tensors; use
                # those (same vocabulary as Google's, different encoding settings)
                def write_asset(key: str, path: str) -> None:
                    with open(path, "wb") as f:
                        f.write(te[key].numpy().tobytes())

                write_asset("hf_asset__generation_config.json", os.path.join(te_dir, "generation_config.json"))
                write_asset("tokenizer_json", os.path.join(tok_dir, "tokenizer.json"))
                write_asset("hf_asset__tokenizer_config.json", os.path.join(tok_dir, "tokenizer_config.json"))
                write_asset("hf_asset__chat_template.jinja", os.path.join(tok_dir, "chat_template.jinja"))
                from transformers import AutoTokenizer

                AutoTokenizer.from_pretrained(tok_dir)
                print("saved text_encoder and tokenizer", flush=True)
            del hf_state
        del te

    if only & {"transformer", "connectors"}:
        dit = prefix(load("dit"), "model.diffusion_model.")
        if "transformer" in only:
            save(conv.convert_ltx2_transformer(dit, version="2.5"), "transformer")
        if "connectors" in only:
            connectors = conv.convert_ltx2_connectors(
                {**dit, **text_projections}, version="2.5", gemma_text_config=gemma_cfg.text_config
            )
            save(connectors, "connectors")
        del dit

    if "scheduler" in only and not args.meta:
        from diffusers import FlowMatchEulerDiscreteScheduler

        FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=False,
            base_shift=0.95,
            max_shift=2.05,
            base_image_seq_len=1024,
            max_image_seq_len=4096,
            shift_terminal=None,
        ).save_pretrained(os.path.join(args.out, "scheduler"))
        print("saved scheduler", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
