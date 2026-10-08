#!/usr/bin/env bash
set -euo pipefail

target="${1:?target is required}"
output="${2:-dist}"
root="$(pwd -P)"
cargo_home="${CARGO_HOME:-$HOME/.cargo}"
if command -v cygpath >/dev/null 2>&1; then
  root="$(cygpath -m "$root")"
  cargo_home="$(cygpath -m "$cargo_home")"
fi
if [[ "$target" == x86_64-pc-windows-msvc ]]; then
  msvc_bin="$(cygpath -u "${VCToolsInstallDir:?Windows build tools are not initialized}")/bin/Hostx64/x64"
  test -f "$msvc_bin/link.exe"
  export PATH="$msvc_bin:$PATH"
  CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER="$(cygpath -m "$msvc_bin/link.exe")"
  export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER
fi
flags=("--remap-path-prefix=$root=/src" "--remap-path-prefix=$cargo_home=/cargo")
if [[ "$target" == x86_64-* ]]; then
  flags+=(-C target-cpu=x86-64-v3)
fi
if [[ "$target" == *-windows-msvc ]]; then
  flags+=(-C target-feature=+crt-static)
fi
printf -v CARGO_ENCODED_RUSTFLAGS '%s\x1f' "${flags[@]}"
export CARGO_ENCODED_RUSTFLAGS="${CARGO_ENCODED_RUSTFLAGS%$'\x1f'}"
unset RUSTFLAGS
export CARGO_INCREMENTAL=0
SOURCE_DATE_EPOCH="$(git -c safe.directory="$root" show -s --format=%ct HEAD)"
export SOURCE_DATE_EPOCH
export MACOSX_DEPLOYMENT_TARGET=11.0
export TZ=UTC
export LC_ALL=C

rustup show
rustup target add "$target"
packages=(-p svpflow1 -p svpflow2 -p svpflow-capi)
cargo build --release --locked --target "$target" "${packages[@]}"

case "$target" in
  *-windows-msvc) ext=dll ;;
  *-apple-darwin) ext=dylib ;;
  *-linux-gnu) ext=so ;;
  *) exit 1 ;;
esac
mkdir "$output"
cp target/"$target"/release/*."$ext" "$output/"
cp LICENSE crates/svpflow-capi/include/open_svpflow.h "$output/"
cargo tree --locked --target "$target" -e normal,build --prefix none --format '{p}' "${packages[@]}" \
  | awk 'NF == 2 { print $1 "-" substr($2, 2) }' | sort -u | while read -r crate; do
  for file in "$cargo_home"/registry/src/*/"$crate"/LICENSE*; do
    printf '%s\n\n' "$crate"
    cat "$file"
    printf '\n\n'
  done
done > "$output/CRATES.txt"
{
  echo "Commit: $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "Target: $target"
  rustc -Vv
} > "$output/BUILD.txt"
