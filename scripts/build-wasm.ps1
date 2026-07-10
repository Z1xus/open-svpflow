$root = Split-Path -Parent $PSScriptRoot
$crate = Join-Path $root "crates\svpflow-wasm"
$out = Join-Path $root "target\wasm"
$env:RUSTUP_TOOLCHAIN = "nightly-2026-07-10"
$env:CARGO_UNSTABLE_BUILD_STD = "panic_abort,std"
$env:RUSTFLAGS = "-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 -C link-arg=--shared-memory -C link-arg=--max-memory=1073741824 -C link-arg=--import-memory -C link-arg=--export=__heap_base -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base"
$packArgs = @("build", $crate, "--target", "web", "--release", "--out-dir", $out)
& wasm-pack @packArgs
if ($LASTEXITCODE -eq 0) {
    Copy-Item (Join-Path $crate "webgpu.js") $out
}
exit $LASTEXITCODE
