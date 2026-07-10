#!/usr/bin/env sh
set -eu

root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
crate="$root/crates/svpflow-wasm"
out="$root/target/wasm"
export RUSTUP_TOOLCHAIN=nightly-2026-07-10
export CARGO_UNSTABLE_BUILD_STD="panic_abort,std"
export RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 -C link-arg=--shared-memory -C link-arg=--max-memory=1073741824 -C link-arg=--import-memory -C link-arg=--export=__heap_base -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base"

wasm-pack build "$crate" --target web --release --out-dir "$out"
cp "$crate/webgpu.js" "$out/webgpu.js"
