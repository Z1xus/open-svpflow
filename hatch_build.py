import os
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PLATFORMS = {
    "x86_64-pc-windows-msvc": "win_amd64",
    "x86_64-unknown-linux-gnu": "manylinux_2_34_x86_64",
    "aarch64-unknown-linux-gnu": "manylinux_2_34_aarch64",
    "aarch64-apple-darwin": "macosx_11_0_arm64",
    "x86_64-apple-darwin": "macosx_11_0_x86_64",
}


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        bin_dir = Path(os.environ["SVPFLOW_BIN"])
        build = (bin_dir / "BUILD.txt").read_text()
        target = next(line.split(": ", 1)[1] for line in build.splitlines() if line.startswith("Target: "))
        libs = sorted(bin_dir.glob("*_vs.*"))
        if len(libs) != 2:
            raise RuntimeError(f"expected 2 plugins in {bin_dir}, found {len(libs)}")
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{PLATFORMS[target]}"
        for lib in libs:
            build_data["force_include"][str(lib)] = f"vapoursynth/plugins/open-svpflow/{lib.name}"
