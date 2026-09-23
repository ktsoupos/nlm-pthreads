#include "patch.h"

float patch_dist2_scalar(const image_t *img, int yi, int xi, int yj, int xj, int patch_radius){
    float sum = 0.0f;
    int patch_size = 2 * patch_radius + 1;

    for(int dyi = 0; dyi < patch_size; dyi++){
        for(int dxi = 0; dxi < patch_size; dxi++){
            float pi = IMG_AT(img, yi + dyi - patch_radius, xi + dxi - patch_radius);
            float pj = IMG_AT(img, yj + dyi - patch_radius, xj + dxi - patch_radius);
            float diff = pi - pj;
            sum += diff * diff;
        }
    }

    return (sum)/(patch_size * patch_size);
}