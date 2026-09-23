#ifndef PATCH_H
#define PATCH_H

#include "image.h"

/**
 * @brief Mean squared difference between the patches centered at (yi,xi)
 *        and (yj,xj).
 *
 * Compares the (2*patch_radius+1)^2 patch around each center pixel and
 * returns the sum of squared differences divided by the patch area — this
 * is d^2(i,j) as defined in CLAUDE.md, not a raw sum.
 *
 * No bounds checking: the caller must ensure both patches, including
 * patch_radius in every direction, stay within img's padded region.
 *
 * @param img           Source image (padded).
 * @param yi, xi         Center of patch i.
 * @param yj, xj         Center of patch j.
 * @param patch_radius  Patch half-width; patch side is 2*patch_radius+1.
 * @return              Mean squared difference between the two patches.
 */
float patch_dist2_scalar(const image_t *img, int yi, int xi, int yj, int xj, int patch_radius);

/**
 * @brief AVX2 implementation of patch_dist2_scalar(). Same signature, same
 *        contract, identical output — only nlm_pthreads_simd calls this one.
 */
float patch_dist2_avx2(const image_t *img, int yi, int xi, int yj, int xj, int patch_radius);

#endif
