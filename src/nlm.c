#include "nlm.h"
#include "patch.h"
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/** Patch-distance backend; scalar and AVX2 share one signature so the only
 *  difference between a variant and its SIMD twin is which one gets passed in. */
typedef float (*patch_dist_fn)(const image_t *, int, int, int, int, int);

/** A body that denoises output rows [y0, y1). Reads only `in` and writes only
 *  those rows of `out`, so disjoint ranges need no synchronisation. */
typedef void (*rows_fn)(const image_t *, image_t *, const nlm_params *,
                        patch_dist_fn, int, int);

static void check_io(const image_t *in, const image_t *out);
static void direct_rows(const image_t *in, image_t *out, const nlm_params *p,
                        patch_dist_fn dist, int y0, int y1);
static void integral_rows(const image_t *in, image_t *out, const nlm_params *p,
                          patch_dist_fn dist, int y0, int y1);
static void run_parallel(const image_t *in, image_t *out, const nlm_params *p,
                         rows_fn body, patch_dist_fn dist);

void nlm_run(const image_t *in, image_t *out, const nlm_params *p, nlm_variant v){
    check_io(in, out);
    switch(v){
        case NLM_SEQ:
            direct_rows(in, out, p, patch_dist2_scalar, 0, in->height); break;
        case NLM_SEQ_SIMD:
            direct_rows(in, out, p, patch_dist2_avx2, 0, in->height); break;
        case NLM_PTHREADS:
            run_parallel(in, out, p, direct_rows, patch_dist2_scalar); break;
        case NLM_PTHREADS_SIMD:
            run_parallel(in, out, p, direct_rows, patch_dist2_avx2); break;
        case NLM_INTEGRAL:
            integral_rows(in, out, p, NULL, 0, in->height); break;
        case NLM_INTEGRAL_PTHREADS:
            run_parallel(in, out, p, integral_rows, NULL); break;
    }
}

static void check_io(const image_t *in, const image_t *out){
    assert(in->width == out->width && in->height == out->height);
    assert(in->pad == out->pad && in->stride == out->stride);
    assert(in->data != out->data);  // denoising in place corrupts patches later pixels still read
    (void)in; (void)out;
}


/* ---------------------------------------------------------------- direct */

static void direct_rows(const image_t *in, image_t *out, const nlm_params *p,
                        patch_dist_fn dist, int y0, int y1){
    for(int yi = y0; yi < y1; yi++){
        for(int xi = 0; xi < in->width; xi++){
            float Z = 0.0f, wsum = 0.0f, max_w = 0.0f;

            for(int yj = yi - p->search_radius; yj <= yi + p->search_radius; yj++){
                for(int xj = xi - p->search_radius; xj <= xi + p->search_radius; xj++){
                    if(yj == yi && xj == xi) continue;

                    float d2 = dist(in, yi, xi, yj, xj, p->patch_radius);
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


/* -------------------------------------------------------------- integral */

/**
 * Integral / separable-box reformulation (Darbon et al.), restricted to output
 * rows [y0, y1).
 *
 * Rather than looping over pixels and then their search window, loop over
 * displacements t = j - i. For a fixed t the squared-difference image is shared
 * by every pixel, so a running box sum answers "patch sum at pixel i" in O(1).
 * That removes the P^2 patch factor: O(N*S^2*P^2) -> O(N*S^2).
 *
 * The box sum is kept separable and rolling rather than as a 2D summed-area
 * table. `col[]` holds, per column, the sum over the patch's 2*pr+1 rows;
 * stepping down a row adds the incoming row and subtracts the outgoing one, and
 * a horizontal running window over col[] then gives each pixel's patch sum.
 * That keeps the working set at one row of floats (a few KB, L1-resident)
 * instead of a full (rows+2pr)x(W+2pr) table of doubles, which is what the
 * thread-scaling measurements showed to be the bottleneck.
 *
 * Shrinking the window also removes the need for double. A 2D table accumulates
 * to ~1.7e10, far past float32's 2^24 exact-integer limit, so patch sums taken
 * as differences of table entries lose precision badly. Here a column sum tops
 * out at (2*pr+1)*255^2 and a patch sum at (2*pr+1)^2*255^2 = 3.19e6 for a 7x7
 * patch -- every intermediate is an exactly representable integer, so the
 * rolling updates never drift and this reproduces direct_rows bit for bit.
 *
 * Each call covers only its own rows, reading a pr-row halo above and below, so
 * parallel row blocks share no accumulator and need no reduction.
 */
static void integral_rows(const image_t *in, image_t *out, const nlm_params *p,
                          patch_dist_fn dist, int y0, int y1){
    (void)dist;   // the integral form has no per-pair patch-distance call to swap

    const int W = in->width;
    const int pr = p->patch_radius, sr = p->search_radius;
    const int area = (2*pr + 1) * (2*pr + 1);
    const float noise_floor = 2.0f * p->sigma * p->sigma;
    const float h2 = p->h * p->h;

    const int rows = y1 - y0;
    const int cw = W + 2*pr;      // columns spanned: image x in [-pr, W-1+pr]

    float *col      = malloc((size_t)cw * sizeof *col);   // col[c] <-> image x = c - pr
    float *acc_z    = calloc((size_t)rows * W, sizeof *acc_z);
    float *acc_wsum = calloc((size_t)rows * W, sizeof *acc_wsum);
    float *acc_maxw = calloc((size_t)rows * W, sizeof *acc_maxw);

    if(!col || !acc_z || !acc_wsum || !acc_maxw){
        free(col); free(acc_z); free(acc_wsum); free(acc_maxw);
        fprintf(stderr, "nlm_integral: out of memory\n");
        exit(EXIT_FAILURE);
    }

    for(int dy = -sr; dy <= sr; dy++){
        for(int dx = -sr; dx <= sr; dx++){
            if(dy == 0 && dx == 0) continue;   // self-weight applied after the loop

            // Seed the column sums for the first output row: rows [y0-pr, y0+pr].
            for(int c = 0; c < cw; c++){
                const int x = c - pr;
                float s = 0.0f;
                for(int y = y0 - pr; y <= y0 + pr; y++){
                    const float d = IMG_AT(in, y, x) - IMG_AT(in, y + dy, x + dx);
                    s += d * d;
                }
                col[c] = s;
            }

            for(int yi = y0; yi < y1; yi++){
                if(yi > y0){
                    // Roll down one row: add row yi+pr, drop row yi-1-pr.
                    const int y_add = yi + pr, y_drop = yi - 1 - pr;
                    for(int c = 0; c < cw; c++){
                        const int x = c - pr;
                        const float a = IMG_AT(in, y_add,  x) - IMG_AT(in, y_add  + dy, x + dx);
                        const float r = IMG_AT(in, y_drop, x) - IMG_AT(in, y_drop + dy, x + dx);
                        col[c] += a*a - r*r;
                    }
                }

                // Horizontal running window: pixel xi covers col[] indices [xi, xi+2pr].
                float running = 0.0f;
                for(int c = 0; c <= 2*pr; c++) running += col[c];

                const size_t base = (size_t)(yi - y0) * W;
                float *z  = &acc_z[base];
                float *ws = &acc_wsum[base];
                float *mw = &acc_maxw[base];

                for(int xi = 0; xi < W; xi++){
                    if(xi > 0) running += col[xi + 2*pr] - col[xi - 1];

                    const float d2 = running / (float)area;
                    const float w  = expf(-fmaxf(d2 - noise_floor, 0.0f) / h2);

                    if(w > mw[xi]) mw[xi] = w;
                    z[xi]  += w;
                    ws[xi] += w * IMG_AT(in, yi + dy, xi + dx);
                }
            }
        }
    }

    for(int yi = y0; yi < y1; yi++){
        const size_t base = (size_t)(yi - y0) * W;
        for(int xi = 0; xi < W; xi++){
            const float centre = IMG_AT(in, yi, xi);
            const float z  = acc_z[base + xi]    + acc_maxw[base + xi];
            const float ws = acc_wsum[base + xi] + acc_maxw[base + xi] * centre;
            IMG_AT(out, yi, xi) = (z > 0.0f) ? (ws / z) : centre;
        }
    }

    free(col); free(acc_z); free(acc_wsum); free(acc_maxw);
}


/* -------------------------------------------------------------- threading */

typedef struct {
    const image_t *in;
    image_t *out;
    const nlm_params *p;
    rows_fn body;
    patch_dist_fn dist;
    int y0, y1;
} worker_arg;

static void *worker_main(void *arg){
    const worker_arg *a = (const worker_arg *)arg;
    a->body(a->in, a->out, a->p, a->dist, a->y0, a->y1);
    return NULL;
}

static void run_parallel(const image_t *in, image_t *out, const nlm_params *p,
                         rows_fn body, patch_dist_fn dist){
    int nthreads = p->threads;
    if(nthreads < 1)          nthreads = 1;
    if(nthreads > in->height) nthreads = in->height;   // never create idle workers

    if(nthreads == 1){
        body(in, out, p, dist, 0, in->height);
        return;
    }

    pthread_t  *tids = malloc((size_t)nthreads * sizeof *tids);
    worker_arg *args = malloc((size_t)nthreads * sizeof *args);
    if(!tids || !args){
        free(tids); free(args);
        body(in, out, p, dist, 0, in->height);   // degrade to serial rather than fail
        return;
    }

    // Contiguous row blocks, not interleaved rows: neighbouring output rows reuse
    // overlapping search windows, so a contiguous block keeps that reuse inside one
    // core's cache. Work per pixel is constant (S^2 * P^2), so static splitting is
    // already balanced. The first (height % nthreads) blocks absorb the remainder.
    const int base = in->height / nthreads;
    const int extra = in->height % nthreads;
    int y = 0;
    for(int t = 0; t < nthreads; t++){
        const int rows = base + (t < extra ? 1 : 0);
        args[t].in = in; args[t].out = out; args[t].p = p;
        args[t].body = body; args[t].dist = dist;
        args[t].y0 = y; args[t].y1 = y + rows;
        y += rows;
    }
    assert(y == in->height);

    int started = 0;
    for(int t = 0; t < nthreads; t++){
        if(pthread_create(&tids[t], NULL, worker_main, &args[t]) != 0) break;
        started++;
    }
    // Any block whose thread could not start runs here, so the output is never
    // left partially denoised.
    for(int t = started; t < nthreads; t++)
        body(in, out, p, dist, args[t].y0, args[t].y1);

    for(int t = 0; t < started; t++)
        pthread_join(tids[t], NULL);

    free(tids);
    free(args);
}
