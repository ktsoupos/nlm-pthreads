#include "nlm.h"
#include "patch.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void nlm_seq(const image_t *in, image_t *out, const nlm_params *p);
static void nlm_pthreads(const image_t *in, image_t *out, const nlm_params *p, nlm_variant v);
static void nlm_integral(const image_t *in, image_t *out, const nlm_params *p);

void nlm_run(const image_t *in, image_t *out, const nlm_params *p, nlm_variant v){
    switch(v){
        case NLM_SEQ:      nlm_seq(in, out, p); break;
        case NLM_PTHREADS:
        case NLM_PTHREADS_SIMD:
                             nlm_pthreads(in, out, p, v); break;
        case NLM_INTEGRAL: nlm_integral(in, out, p); break;
    }
}


static void nlm_seq(const image_t *in, image_t *out, const nlm_params *p){
    assert(in->width == out->width && in->height == out->height);
    assert(in->pad == out->pad && in->stride == out->stride);
    assert(in->data != out->data);  // denoising in place corrupts patches later pixels still read

    for(int yi = 0; yi < in->height; yi++){
        for(int xi = 0; xi < in->width; xi++){
            float Z = 0.0f, wsum = 0.0f, max_w = 0.0f;

            for(int yj = yi - p->search_radius; yj <= yi + p->search_radius; yj++){
                for(int xj = xi - p->search_radius; xj <= xi + p->search_radius; xj++){
                    if(yj == yi && xj == xi) continue;

                    float d2 = patch_dist2_scalar(in, yi, xi, yj, xj, p->patch_radius);
                    float w  = expf(-fmaxf(d2 - 2.0f*p->sigma*p->sigma, 0.0f) / (p->h*p->h));

                    if(w > max_w) max_w = w;
                    Z    += w;
                    wsum += w * IMG_AT(in, yj, xj);
                }
            }

            Z    += max_w;
            wsum += max_w * IMG_AT(in, yi, xi);

            // Every weight can underflow to zero (small h on high-contrast content,
            // or search_radius == 0), leaving Z == 0. No patch supports an estimate
            // here, so the pixel's own value is the best one available.
            IMG_AT(out, yi, xi) = (Z > 0.0f) ? (wsum / Z) : IMG_AT(in, yi, xi);
        }
    }
}






static void nlm_pthreads(const image_t *in, image_t *out, const nlm_params *p, nlm_variant v){
    (void)in; (void)out; (void)p; (void)v;
    fprintf(stderr, "nlm: NLM_PTHREADS / NLM_PTHREADS_SIMD not implemented yet\n");
    exit(EXIT_FAILURE);
}

static void nlm_integral(const image_t *in, image_t *out, const nlm_params *p){
    (void)in; (void)out; (void)p;
    fprintf(stderr, "nlm: NLM_INTEGRAL not implemented yet\n");
    exit(EXIT_FAILURE);
}