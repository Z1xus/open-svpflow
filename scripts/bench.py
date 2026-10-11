import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

REFINE = "refine:[{thsad:250,search:{distance:-1,satd:true}}]"
MEDIUM = "main:{search:{distance:0,coarse:{distance:-1,trymany:true,bad:{sad:2000}}}}"
FASTER = "main:{search:{coarse:{distance:-1,trymany:true,bad:{sad:2000}}}}"
PRESETS = {
    ("faster", "cpu"): ("{pel:1,gpu:0}", f"{{block:{{w:16,overlap:2}},{FASTER}}}"),
    ("faster", "gpu"): ("{pel:1,gpu:1}", f"{{block:{{w:16,overlap:1}},{FASTER}}}"),
    ("medium", "cpu"): ("{gpu:0}", f"{{block:{{w:16,overlap:2}},{MEDIUM},{REFINE}}}"),
    ("medium", "gpu"): ("{gpu:1}", f"{{block:{{w:8,overlap:2}},{MEDIUM},{REFINE}}}"),
}
SMOOTH = (
    "{{rate:{{num:{fps},den:1,abs:true}},algo:23,mask:{{cover:80,area:0,area_sharp:1.2}},"
    "scene:{{blend:true,mode:0,limits:{{blocks:50}}}}}}"
)


def decode(core, vs, path, count):
    probe = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "csv=p=0"]
    probe += ["-show_entries", "stream=width,height,r_frame_rate", path]
    width, height, rate = (
        subprocess.check_output(probe, text=True).strip().split(",")[:3]
    )
    width, height = int(width), int(height)
    num, den = map(int, rate.split("/"))
    pipe = ["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(count)]
    pipe += ["-pix_fmt", "yuv420p", "-f", "rawvideo", "-"]
    data = subprocess.check_output(pipe)
    size = width * height * 3 // 2
    blank = core.std.BlankClip(
        format=vs.YUV420P8,
        width=width,
        height=height,
        fpsnum=num,
        fpsden=den,
        length=len(data) // size,
        keep=True,
    )
    frames = []
    for index in range(blank.num_frames):
        frame, offset = blank.get_frame(0).copy(), index * size
        for plane in range(3):
            w, h = (width, height) if plane == 0 else (width // 2, height // 2)
            target, stride = frame.get_write_ptr(plane).value, frame.get_stride(plane)
            for row in range(h):
                ctypes.memmove(target + row * stride, data[offset : offset + w], w)
                offset += w
        frames.append(frame)
    return core.std.ModifyFrame(blank, blank, lambda n, f: frames[n])


def measure(args):
    import vapoursynth as vs

    core = vs.core
    for name in ("svpflow1", "svpflow2"):
        libraries = Path(args.plugins[0]).glob(f"*{name}*")
        core.std.LoadPlugin(
            next(str(p) for p in libraries if p.suffix in (".so", ".dll", ".dylib"))
        )
    clip = decode(core, vs, args.clip, args.frames) * args.loops
    preset, path = args.run
    super_options, analyse_options = PRESETS[preset, path]
    sup = core.svp1.Super(clip, super_options)
    vectors = core.svp1.Analyse(sup["clip"], sup["data"], clip, analyse_options)
    out = core.svp2.SmoothFps(
        clip,
        sup["clip"],
        sup["data"],
        vectors["clip"],
        vectors["data"],
        SMOOTH.format(fps=args.fps),
    )
    begin = mark = now = time.perf_counter()
    count = 0
    for _ in out.frames(prefetch=args.prefetch, close=True):
        now = time.perf_counter()
        if now - begin < args.warmup:
            mark = now
            continue
        count += 1
        if now - mark >= args.seconds:
            break
    print(count / (now - mark), flush=True)
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Compare the speed of SVPflow builds on one clip."
    )
    parser.add_argument("clip", help="8-bit 4:2:0 video, read with ffmpeg")
    parser.add_argument(
        "plugins", nargs="+", help="directory with svpflow1 and svpflow2"
    )
    parser.add_argument("--fps", type=int, default=1920, help="output frame rate")
    parser.add_argument("--frames", type=int, default=200, help="source frames to use")
    parser.add_argument("--loops", type=int, default=200, help="times to repeat them")
    parser.add_argument("--seconds", type=float, default=10, help="timed part of a run")
    parser.add_argument(
        "--warmup", type=float, default=3, help="untimed start of a run"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--prefetch", type=int, default=48, help="frames requested at once"
    )
    parser.add_argument("--run", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run:
        return measure(args)
    passed = [
        f"--{name}={getattr(args, name)}"
        for name in ("fps", "frames", "loops", "seconds", "warmup", "prefetch")
    ]
    best = {}
    for _ in range(args.rounds):
        for run in PRESETS:
            for plugins in args.plugins:
                command = [
                    sys.executable,
                    __file__,
                    args.clip,
                    plugins,
                    "--run",
                    *run,
                    *passed,
                ]
                fps = float(subprocess.check_output(command, text=True).split()[-1])
                best[run, plugins] = max(best.get((run, plugins), 0), fps)
                print(*run, plugins, f"{fps:.1f}", file=sys.stderr)
    names = [Path(p).resolve().name for p in args.plugins]
    print("| Preset | Path |", " | ".join(names), "| Speed |")
    print("| --- | --- |", " | ".join("---:" for _ in names), "| ---: |")
    for run in PRESETS:
        speeds = [best[run, plugins] for plugins in args.plugins]
        cells = " | ".join(f"{fps:.1f} fps" for fps in speeds)
        print(
            f"| {run[0]} | {run[1].upper()} | {cells} | {speeds[-1] / speeds[0]:.2f}x |"
        )


if __name__ == "__main__":
    main()
