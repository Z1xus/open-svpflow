pub(crate) const VERSION: i64 = 1_174_405_393;

pub(crate) const PLUGIN_ID: &[u8] = b"com.svp-team.flow2\0";
pub(crate) const PLUGIN_NS: &[u8] = b"svp2\0";
pub(crate) const PLUGIN_NAME: &[u8] = b"SVPFlow2\0";

pub(crate) const FILTER_NAME: &[u8] = b"SVSmoothFps\0";
pub(crate) const SMOOTH_FPS: &[u8] = b"SmoothFps\0";
pub(crate) const SMOOTH_FPS_NVOF: &[u8] = b"SmoothFps_NVOF\0";
pub(crate) const SMOOTH_FPS_RIFE: &[u8] = b"SmoothFps_RIFE\0";

pub(crate) const ARGS_SMOOTH_FPS: &[u8] =
    b"clip:clip;super:clip;sdata:int;vectors:clip;vdata:int;opt:data;src:clip:opt;fps:float:opt\0";
pub(crate) const ARGS_NVOF: &[u8] =
    b"clip:clip;opt:data;vec_src:clip:opt;src:clip:opt;fps:float:opt\0";
pub(crate) const ARGS_RIFE: &[u8] =
    b"clip:clip;opt:data;rife_out:clip;vec_src:clip:opt;vdata:int:opt;src:clip:opt;fps:float:opt\0";

pub(crate) const ERR_VSAPI: &[u8] = b"SVSmoothFps: invalid VSAPI table\0";
pub(crate) const ERR_FPS: &[u8] = b"SVSmoothFps: unable to determine source frame rate\0";
pub(crate) const ERR_PARAMS: &[u8] = b"SVSmoothFps: invalid 'params' syntax: \0";
pub(crate) const ERR_ALGO: &[u8] = b"SVSmoothFps: incorrect 'algo' value\0";
pub(crate) const ERR_CUBIC: &[u8] =
    b"SVSmoothFps: 'cubic' mode isn't available with CPU rendering\0";
pub(crate) const ERR_SCENE_MODE: &[u8] = b"SVSmoothFps: 'scene.mode' must be in [0;3]\0";
pub(crate) const ERR_SCENE_MODE_RATE: &[u8] =
    b"SVSmoothFps: 'scene.mode' value isn't supported for selected frame rate\0";
pub(crate) const ERR_SOURCE_CPU: &[u8] =
    b"SVSmoothFps: source must be YV12 (8-bit 4:2:0) for CPU rendering\0";
pub(crate) const ERR_SOURCE_YUV: &[u8] = b"SVSmoothFps: source must be YUV 4:2:0 8/10/16-bits\0";
pub(crate) const ERR_VECTORS_SIZE: &[u8] =
    b"SVSmoothFps: source and vectors frame sizes are different\0";
pub(crate) const ERR_VECTORS_INVALID_1: &[u8] = b"SVSmoothFps: invalid vectors stream [1]\0";
pub(crate) const ERR_VECTORS_INVALID_2: &[u8] = b"SVSmoothFps: invalid vectors stream [2]\0";
pub(crate) const ERR_SUPER_PARAMS: &[u8] = b"SVSmoothFps: invalid super clip params\0";
pub(crate) const ERR_VECTORS_DELTA: &[u8] = b"SVSmoothFps: vectors with delta>1 are not allowed.\0";
pub(crate) const ERR_OVERLAP_CPU: &[u8] = b"SVSmoothFps: overlap must be even with CPU rendering\0";
pub(crate) const ERR_VEC_SRC_REQUIRED: &[u8] =
    b"SVSmoothFps/NVOF: 8-bit 'vec_src' must be defined for 10-bit rendering\0";
pub(crate) const ERR_VEC_SRC_FORMAT: &[u8] =
    b"SVSmoothFps/NVOF: 'vec_src' must be YV12 8-bit with the same length as the source\0";
pub(crate) const ERR_VEC_SRC_RATIO: &[u8] =
    b"SVSmoothFps/NVOF: 'vec_src' must be in [1/1,1/2,1/4,1/6,1/8] of the source size\0";
pub(crate) const ERR_VEC_SRC_BLOCKS: &[u8] =
    b"SVSmoothFps/NVOF: minimal 4*4 blocks amount is 40*32\0";
pub(crate) const ERR_CUDA_UNAVAILABLE: &[u8] = b"SVSmoothFps: CUDA is not availabe\0";
pub(crate) const ERR_NVOF_INIT: &[u8] = b"SVSmoothFps: unable to init NVOF - code \0";
pub(crate) const ERR_GPU_INIT: &[u8] = b"SVSmoothFps: unable to init GPU-based renderer - code \0";
pub(crate) const ERR_GPU_FALLBACK: &[u8] = b", falling back to CPU mode\0";

pub(crate) const CLIP: &[u8] = b"clip\0";
pub(crate) const FPS: &[u8] = b"fps\0";
pub(crate) const OPT: &[u8] = b"opt\0";
pub(crate) const RIFE_OUT: &[u8] = b"rife_out\0";
pub(crate) const SDATA: &[u8] = b"sdata\0";
pub(crate) const SRC: &[u8] = b"src\0";
pub(crate) const SUPER: &[u8] = b"super\0";
pub(crate) const VDATA: &[u8] = b"vdata\0";
pub(crate) const VEC_SRC: &[u8] = b"vec_src\0";
pub(crate) const VECTORS: &[u8] = b"vectors\0";

pub(crate) const DURATION_NUM: &[u8] = b"_DurationNum\0";
pub(crate) const DURATION_DEN: &[u8] = b"_DurationDen\0";
pub(crate) const PTS: &[u8] = b"_PTS\0";
pub(crate) const DOVI: &[u8] = b"_DoVi\0";
pub(crate) const SCENE_CHANGE_NEXT: &[u8] = b"_SceneChangeNext\0";
