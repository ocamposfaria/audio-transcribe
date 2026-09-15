#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["faster-whisper"]
# ///
"""Transcreve arquivos de audio (.m4a, .mp3, .wav, ...) localmente com faster-whisper.

Uso:
    uv run transcribe.py AUDIO [-l pt] [-m large-v3] [-o saida.txt] [--timestamps]
"""
import argparse
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Transcricao local de audio com faster-whisper")
    ap.add_argument("audio", type=Path)
    ap.add_argument("-l", "--language", default="pt", help="codigo do idioma (default: pt)")
    ap.add_argument("-m", "--model", default="large-v3", help="modelo whisper (default: large-v3)")
    ap.add_argument("-o", "--output", type=Path, help="arquivo de saida (default: stdout)")
    ap.add_argument("--timestamps", action="store_true", help="prefixar cada trecho com [hh:mm:ss]")
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"arquivo nao encontrado: {args.audio}", file=sys.stderr)
        return 1

    print(f"carregando modelo {args.model}...", file=sys.stderr)
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        str(args.audio),
        language=args.language,
        vad_filter=True,
        beam_size=5,
    )
    print(f"idioma: {info.language} | duracao: {fmt_ts(info.duration)}", file=sys.stderr)

    lines = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        lines.append(f"[{fmt_ts(seg.start)}] {text}" if args.timestamps else text)
        print(".", end="", flush=True, file=sys.stderr)
    print("", file=sys.stderr)

    out = "\n".join(lines) if args.timestamps else " ".join(lines)
    if args.output:
        args.output.write_text(out + "\n", encoding="utf-8")
        print(f"salvo em {args.output}", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
