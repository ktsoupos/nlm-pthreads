#ifndef PATCH_H
#define PATCH_H

#include "image.h"

/*
 * Mean squared difference between the patches centred at (yi,xi) and (yj,xj) --
 * the sum divided by the patch area, not the raw sum.
 *
 * No bounds checking: both patches, plus patch_radius in every direction, must
 * lie inside the image's padded region.
 *
 * The two implementations share this signature and must agree exactly.
 */
float patch_dist2_scalar(const image_t *img, int yi, int xi, int yj, int xj, int patch_radius);
float patch_dist2_avx2  (const image_t *img, int yi, int xi, int yj, int xj, int patch_radius);

#endif
