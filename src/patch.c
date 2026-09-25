#include "patch.h"

#include <immintrin.h>

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

static inline float hsum256_ps(__m256 v){
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

float patch_dist2_avx2(const image_t *img, int yi, int xi, int yj, int xj, int patch_radius){
    const int patch_size = 2 * patch_radius + 1;
    const int vec_end    = (patch_size / 8) * 8;   // floats covered by full 8-wide loads
    const int rem        = patch_size - vec_end;

    // Lane k enabled iff k < rem. maskload zeroes disabled lanes and does not
    // access that memory, so a short patch row never reads past the border.
    const __m256i tail_mask = _mm256_cmpgt_epi32(
        _mm256_set1_epi32(rem),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));

    __m256 acc = _mm256_setzero_ps();

    for(int dy = -patch_radius; dy <= patch_radius; dy++){
        const float *pi = &IMG_AT(img, yi + dy, xi - patch_radius);
        const float *pj = &IMG_AT(img, yj + dy, xj - patch_radius);

        int k = 0;
        for(; k < vec_end; k += 8){
            __m256 d = _mm256_sub_ps(_mm256_loadu_ps(pi + k), _mm256_loadu_ps(pj + k));
            acc = _mm256_fmadd_ps(d, d, acc);
        }
        if(rem){
            __m256 d = _mm256_sub_ps(_mm256_maskload_ps(pi + k, tail_mask),
                                     _mm256_maskload_ps(pj + k, tail_mask));
            acc = _mm256_fmadd_ps(d, d, acc);
        }
    }

    return hsum256_ps(acc) / (float)(patch_size * patch_size);
}
