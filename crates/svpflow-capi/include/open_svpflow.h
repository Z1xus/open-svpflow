#ifndef OPEN_SVPFLOW_H
#define OPEN_SVPFLOW_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct osvp_context osvp_context;
typedef struct osvp_clip osvp_clip;
typedef struct osvp_frame osvp_frame;

enum {
    OSVP_YUV420P8,
    OSVP_YUV444P8,
    OSVP_YUV420P10,
    OSVP_YUV420P16,
    OSVP_YUV444P16
};

typedef struct {
    int32_t format;
    int32_t width;
    int32_t height;
    int64_t fps_num;
    int64_t fps_den;
    int32_t num_frames;
} osvp_video_info;

typedef int32_t (*osvp_read)(void *user, int32_t n, uint8_t *const planes[3],
                             const ptrdiff_t strides[3]);
typedef void (*osvp_release)(void *user);
typedef int32_t (*osvp_need)(void *user, int32_t n, int32_t *frames);
typedef int32_t (*osvp_render)(void *user, int32_t n, const osvp_frame *const *inputs,
                               int32_t count, uint8_t *const planes[3],
                               const ptrdiff_t strides[3]);

const char *osvp_error(void);

osvp_context *osvp_create(const char *plugin_dir);
void osvp_destroy(osvp_context *context);

osvp_clip *osvp_source(const osvp_context *context, const osvp_video_info *info, osvp_read read,
                       osvp_release release, void *user);
/* need writes the input frames for frame n, at most 8, and returns their count */
osvp_clip *osvp_map(const osvp_context *context, const osvp_video_info *info,
                    const osvp_clip *input, osvp_need need, osvp_render render,
                    osvp_release release, void *user);

osvp_clip *osvp_smooth_fps(const osvp_context *context, const osvp_clip *source,
                           const char *super_opt, const char *analyse_opt,
                           const char *smooth_opt, int32_t half_analysis);

osvp_clip *osvp_smooth_fps_blend(const osvp_context *context, const osvp_clip *source,
                                 const char *super_opt, const char *analyse_opt,
                                 const char *smooth_opt, int32_t half_analysis,
                                 const double *weights, int32_t weight_count, int64_t fps_num,
                                 int64_t fps_den, double gamma);

osvp_clip *osvp_still(const osvp_context *context, const osvp_clip *clip,
                      const osvp_clip *source, double limit, double edge, double tolerance);

osvp_clip *osvp_layer(const osvp_context *context, const osvp_clip *source,
                      const char *super_opt, const char *analyse_opt, int32_t half_analysis,
                      double limit);

osvp_clip *osvp_peel(const osvp_context *context, const osvp_clip *source,
                     const osvp_clip *layer);

osvp_clip *osvp_unpeel(const osvp_context *context, const osvp_clip *clip,
                       const osvp_clip *source, const osvp_clip *layer);

void osvp_clip_info(const osvp_clip *clip, osvp_video_info *info);
void osvp_clip_free(osvp_clip *clip);

void osvp_clip_set_end(const osvp_clip *clip, int32_t last);
const osvp_frame *osvp_get_frame(const osvp_clip *clip, int32_t n);
const uint8_t *osvp_frame_data(const osvp_frame *frame, int32_t plane);
ptrdiff_t osvp_frame_stride(const osvp_frame *frame, int32_t plane);
void osvp_frame_free(const osvp_frame *frame);

#ifdef __cplusplus
}
#endif

#endif
