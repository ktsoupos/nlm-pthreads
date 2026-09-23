#ifndef NLM_H
#define NLM_H

#include "image.h"

/** @brief Selects which nlm_run() implementation runs; all must produce
 *         identical output — only speed differs. */
typedef enum { NLM_SEQ, NLM_PTHREADS, NLM_PTHREADS_SIMD, NLM_INTEGRAL } nlm_variant;

/**
 * @brief Parameters shared by every NLM variant.
 */
typedef struct {
    int patch_radius;    /**< Patch half-width; patch side is 2*patch_radius+1. */
    int search_radius;   /**< Search-window half-width around each output pixel. */
    float sigma;          /**< Estimated noise standard deviation. */
    float h;                /**< Filtering strength; scale with sigma (h ~= 0.4*sigma for 7x7 patches). */
    int threads;          /**< Worker thread count; ignored by NLM_SEQ. */
} nlm_params;

/**
 * @brief Denoise @p in into @p out using the given variant.
 *
 * @p in and @p out must be distinct images (never denoise in place — later
 * output pixels still need to read @p in's original, unmodified values).
 * Both must be padded by at least @c p->patch_radius + p->search_radius.
 *
 * @param in   Source image, read-only.
 * @param out  Destination image; only its logical region is written.
 * @param p    Algorithm parameters.
 * @param v    Which implementation to run.
 */
void nlm_run(const image_t *in, image_t *out, const nlm_params *p, nlm_variant v);

#endif
