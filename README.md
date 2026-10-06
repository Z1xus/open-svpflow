# open-svpflow

<a href="https://pypi.org/project/open-svpflow/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.shieldcn.dev/pypi/v/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=dark"><img alt="PyPI version" src="https://www.shieldcn.dev/pypi/v/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=light"></picture></a>
<a href="https://pypistats.org/packages/open-svpflow"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.shieldcn.dev/pypi/dm/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=dark"><img alt="PyPI downloads" src="https://www.shieldcn.dev/pypi/dm/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=light"></picture></a>
<a href="https://github.com/Z1xus/open-svpflow/actions/workflows/nightly-release.yml"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.shieldcn.dev/github/ci/Z1xus/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=dark"><img alt="CI status" src="https://www.shieldcn.dev/github/ci/Z1xus/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=light"></picture></a>
<a href="https://github.com/Z1xus/open-svpflow/releases"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.shieldcn.dev/github/downloads/Z1xus/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=dark"><img alt="Total downloads" src="https://www.shieldcn.dev/github/downloads/Z1xus/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=light"></picture></a>
<a href="https://github.com/Z1xus/open-svpflow/commits/main"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.shieldcn.dev/github/last-commit/Z1xus/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=dark"><img alt="Last commit" src="https://www.shieldcn.dev/github/last-commit/Z1xus/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=light"></picture></a>

open-svpflow is a drop-in open-source replacement for SVPFlow's VapourSynth plugins in Rust. It produces the same frames as SVPFlow (matched to version 4.3.0.168) and runs faster on both CPU and GPU.

It started as a side project, mostly to see if I could get couler's [smoothie-rs](https://github.com/couleur-tweak-tips/smoothie-rs) running in a browser. The [demo](https://smoothie.z1x.us) is the result:

[![open-svpflow running in the browser](https://smoothie.z1x.us/demo.webp)](https://smoothie.z1x.us)

This proof of concept runs open-svpflow client-side in the browser as WASM and renders with WebGPU, using [WebCodecs](https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API). The browser version runs about 2-3x slower than the native plugins, since WASM can't use AVX2 or AVX-512. If it's a lot slower than that for you, check `chrome://gpu` (or `about:support` for Firefox) and make sure hardware acceleration and WebGPU are enabled, otherwise it falls back to rendering on the CPU.

## What's implemented

- `svpflow1`: `Super` and `Analyse`, including super-frame pyramids, multi-level predictors, SAD/SATD and chroma costs, and hex2, UMH, and exhaustive motion search.
- `svpflow2`: `SmoothFps` rendering for algorithms 1, 2, 11, 13, 21, 22, and 23, with scene handling, masks, CPU rendering, and the GPU path.
- Native I444 input and output, which the original SVPFlow doesn't support.
- VapourSynth API 4 support, so it works on current VapourSynth (R80 and newer can't load the original at all, since it only has API 3).
- AviSynth+ support with the original function names.
- `SmoothFpsBlend`, an extra function that interpolates and frame-blends on the GPU in one go, so it's about 2x faster than blending separately (build with `--no-default-features` if you don't want it).
- `Halve` and `DoubleVectors`, two more extra functions that let the motion search run on a half-size frame, which is a lot faster for a small quality loss.
- A C API for running it without VapourSynth or AviSynth (`open_svpflow.h`, build it with `cargo build --release -p svpflow-capi`).

Output matches the original byte for byte in everything I've tested, on both CPU and GPU. But GPU algorithms 1 and 2 are the exception, since the original reads uninitialized memory there and doesn't even match its own previous run (±1 on about 0.01% of pixels, same as ours). So it should work anywhere SVPFlow already does, whether that's smoothie, SVP itself, mpv or your own scripts.

Benchmarked on a 1080p 200 fps Counter-Strike 2 clip interpolated to 1920 fps through [smoothie's](https://github.com/couleur-tweak-tips/smoothie-rs) VapourSynth script (Ryzen 5 7600X, RTX 4060, Linux):

| Preset | Path | Original | open-svpflow | Speed vs original | Output |
| --- | --- | ---: | ---: | ---: | --- |
| `faster` | CPU | 305.2 fps | 819.2 fps | 2.68x | identical |
| `faster` | GPU | 1110.8 fps | 1660.5 fps | 1.49x | identical |
| `medium` (default) | CPU | 225.1 fps | 535.7 fps | 2.38x | identical |
| `medium` (default) | GPU | 310.2 fps | 602.8 fps | 1.94x | identical |

The speedup mainly comes from AVX2/AVX-512 motion search, running part of that search on the GPU (when using GPU mode), and caching work the original redoes for every frame.

## Install

```sh
pip install open-svpflow
```

It install with an assumption that you already have VapourSynth installed in the same Python environment (we deliberately not include it as a dependency). Wheels are available for Windows x64, Linux x64 and arm64, and macOS x64 and arm64.

## AviSynth+

The plugins also load in AviSynth+ 3.6 or newer:

```avs
LoadPlugin("svpflow1_vs.dll")
LoadPlugin("svpflow2_vs.dll")
super = SVSuper("{gpu:1}")
vectors = SVAnalyse(super, "{}")
SVSmoothFps(super, vectors, "{rate:{num:5,den:2}}", mt=8)
```

## Releases

Download latest nightly builds for Windows, Linux, and macOS from [Releases](https://github.com/Z1xus/open-svpflow/releases). Versioned releases follow semver and are published to [PyPI](https://pypi.org/project/open-svpflow/).

New releases are reproducible and immutable once published.

To reproduce the binaries, check out the release commit in a clean directory and use the matching build environment from the [build workflow](.github/workflows/build.yml). Run `bash .github/build.sh <target>` with the target from the release's `BUILD.txt`, then compare the files in `dist/` with the extracted release.

## Build

Rust 1.88 or newer:

```sh
cargo build --release -p svpflow1 -p svpflow2
```

The plugins end up in `target/release/`: `svpflow1_vs.dll` and `svpflow2_vs.dll` on Windows, `libsvpflow1_vs.so` and `libsvpflow2_vs.so` on Linux, `.dylib` on macOS.

Licensed under Apache-2.0.
