#ifndef NLM_H
#define NLM_H

#include "image.h"

/* All variants must produce identical output; only speed differs. */
typedef enum {
    NLM_SEQ, NLM_SEQ_SIMD,
    NLM_PTHREADS, NLM_PTHREADS_SIMD,
    NLM_INTEGRAL, NLM_INTEGRAL_PTHREADS
} nlm_variant;

typedef struct {
    int patch_radius;    /* patch side is 2*patch_radius+1 */
    int search_radius;
    float sigma;
    float h;             /* filter strength; scale with sigma */
    int threads;         /* ignored by the non-threaded variants */
} nlm_params;

/*
 * `in` and `out` must be distinct images with matching geometry, padded by at
 * least patch_radius + search_radius. Denoising in place would corrupt patches
 * that later output pixels still need to read.
 */
void nlm_run(const image_t *in, image_t *out, const nlm_params *p, nlm_variant v);

#endif
