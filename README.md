# open-svpflow

<a href="https://pypi.org/project/open-svpflow/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.shieldcn.dev/pypi/v/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=dark"><img alt="PyPI version" src="https://www.shieldcn.dev/pypi/v/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=light"></picture></a>
<a href="https://pypistats.org/packages/open-svpflow"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.shieldcn.dev/pypi/dm/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=dark"><img alt="PyPI downloads" src="https://www.shieldcn.dev/pypi/dm/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=light"></picture></a>
<a href="https://github.com/Z1xus/open-svpflow/actions/workflows/nightly-release.yml"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.shieldcn.dev/github/ci/Z1xus/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=dark"><img alt="CI status" src="https://www.shieldcn.dev/github/ci/Z1xus/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=light"></picture></a>
<a href="https://github.com/Z1xus/open-svpflow/releases"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.shieldcn.dev/github/downloads/Z1xus/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=dark"><img alt="Total downloads" src="https://www.shieldcn.dev/github/downloads/Z1xus/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=light"></picture></a>
<a href="https://github.com/Z1xus/open-svpflow/commits/main"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.shieldcn.dev/github/last-commit/Z1xus/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=dark"><img alt="Last commit" src="https://www.shieldcn.dev/github/last-commit/Z1xus/open-svpflow.svg?variant=secondary&amp;size=xs&amp;mode=light"></picture></a>

open-svpflow is a drop-in open-source replacement for [SVPFlow](https://www.svp-team.com/wiki/Plugins:_SVPflow)'s VapourSynth plugins in Rust, reverse engineered from version 4.3.0.168. It runs up to 2.7x faster on CPU and up to 1.9x faster on GPU, adds [a few things](#on-top-of-svpflow) the original doesn't have, and works in places where the original doesn't.

It started as a side project, mostly to see if I could get couleur's [smoothie-rs](https://github.com/couleur-tweak-tips/smoothie-rs) running in a browser. That became the [demo](https://smoothie.z1x.us), which runs open-svpflow client-side as WASM with WebGPU (about 2-3x slower than native, since WASM can't use AVX2). It was only meant as a proof of concept, but it grew into another project, [interpolini](https://github.com/Z1xus/interpolini), which is meant as a showcase of how much faster you can get without VapourSynth and Python in the way.

## Parity with SVPFlow

- [svpflow1](crates/svpflow1): Super and Analyse, including super-frame pyramids, multi-level predictors, SAD/SATD and chroma costs, and hex2, UMH, and exhaustive motion search.
- [svpflow2](crates/svpflow2): SmoothFps rendering for algorithms 1, 2, 11, 13, 21, 22, and 23, with scene handling, masks, CPU rendering, and the GPU path.
- [AviSynth+](#avisynth) support with the original function names.

Output matches the original byte for byte in everything I've tested, on both CPU and GPU. GPU algorithms 1 and 2 are the exception, since the original reads uninitialized memory there and doesn't even match its own previous run (±1 on about 0.01% of pixels, same as ours). So it should work anywhere SVPFlow already does, whether that's [smoothie](https://github.com/couleur-tweak-tips/smoothie-rs), [SVP](https://www.svp-team.com) itself, [mpv](https://mpv.io) or your own scripts.

Benchmarked on a 1080p 200 fps Counter-Strike 2 clip interpolated to 1920 fps through [smoothie's](https://github.com/couleur-tweak-tips/smoothie-rs) VapourSynth script (Ryzen 5 7600X, RTX 4060, Linux):

| Preset | Path | Original | open-svpflow | Speed vs original |
| --- | --- | ---: | ---: | ---: |
| faster | CPU | 305.2 fps | 819.2 fps | 2.68x |
| faster | GPU | 1110.8 fps | 1660.5 fps | 1.49x |
| medium (default) | CPU | 225.1 fps | 535.7 fps | 2.38x |
| medium (default) | GPU | 310.2 fps | 602.8 fps | 1.94x |

The speedup mainly comes from AVX2/AVX-512 motion search, running part of that search on the GPU (when using GPU mode), and caching work the original redoes for every frame.

## On top of SVPFlow

- [VapourSynth](https://www.vapoursynth.com) API 4 support, so it works on current VapourSynth (R80 and newer can't load the original at all, since it only has API 3).
- Native I444 input and output.
- SmoothFpsBlend, which interpolates and frame-blends on the GPU in one go, so it's about 2x faster than blending separately (build with --no-default-features if you don't want it).
- Halve and DoubleVectors, which let the motion search run on a half-size frame, a lot faster for a small quality loss.
- mask:{still:true} in the SmoothFps options, which finds text, logos and other things that stay in place and keeps them from smearing (GPU only, off by default). It can be tuned with mask:{still:{limit:4.5,edge:5,tolerance:30}}, and svp2.Still(clip, source) does the same for frames from any other interpolator.
- A [C API](crates/svpflow-capi/include/open_svpflow.h) for running it without VapourSynth or AviSynth (cargo build --release -p svpflow-capi).

## Install

```sh
pip install open-svpflow
```

It installs with an assumption that you already have VapourSynth installed in the same Python environment (we deliberately not include it as a dependency). Wheels are available for Windows x64, Linux x64 and arm64, and macOS x64 and arm64.

## AviSynth

The plugins also load in [AviSynth+](https://avs-plus.net) 3.6 or newer:

```avs
LoadPlugin("svpflow1_vs.dll")
LoadPlugin("svpflow2_vs.dll")
super = SVSuper("{gpu:1}")
vectors = SVAnalyse(super, "{}")
SVSmoothFps(super, vectors, "{rate:{num:5,den:2}}", mt=8)
```

## Releases

Download the latest nightly builds for Windows, Linux, and macOS from [Releases](https://github.com/Z1xus/open-svpflow/releases). Versioned releases follow semver and are published to [PyPI](https://pypi.org/project/open-svpflow/).

Releases are reproducible and immutable once published. To reproduce the binaries, check out the release commit in a clean directory and use the matching build environment from the [build workflow](.github/workflows/build.yml). Run [build.sh](.github/build.sh) with the target from the release's BUILD.txt, then compare the files in dist with the extracted release.

## Build

[Rust](https://rustup.rs) 1.88 or newer:

```sh
cargo build --release -p svpflow1 -p svpflow2
```

The plugins end up in target/release: svpflow1_vs.dll and svpflow2_vs.dll on Windows, libsvpflow1_vs.so and libsvpflow2_vs.so on Linux, .dylib on macOS.

Licensed under Apache-2.0.
