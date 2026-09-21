#include <stdio.h>
#include <stdlib.h>

#include "image.h"
#include "stb_image.h"
#include "stb_image_write.h"

// 64 = L1/L2 cache line size on the target CPU; keeping every row aligned to
// it lets AVX2 loads use the faster aligned form instead of always falling
// back to unaligned loads.
#define ALIGN_BYTES 64
#define ALIGN_FLOATS (ALIGN_BYTES / (int)sizeof(float))


image_t image_load(const char *path, int pad){
    image_t img = {0};
    unsigned char *data = stbi_load(path, &img.width, &img.height, NULL, 1);
    if(NULL == data){
        return img;
    }

    img.pad = pad;
    img.stride = (img.width + 2*pad + ALIGN_FLOATS - 1) & ~(ALIGN_FLOATS - 1);
    img.data = aligned_alloc(ALIGN_BYTES, img.stride * (img.height + 2*pad) * sizeof(float));
    if(NULL == img.data){
        stbi_image_free(data);
        return img;
    }

    for(int y = 0; y < img.height; y++){
        for(int x = 0; x < img.width; x++){
            IMG_AT(&img, y, x) = (float)data[y * img.width + x];
        }
    }

    for(int y = 0; y < img.height; y++){
        for (int x = 0; x < img.pad; x++){
            IMG_AT(&img, y, -x-1) = IMG_AT(&img, y, x+1);
            IMG_AT(&img, y, img.width+x) = IMG_AT(&img, y, img.width-2-x);
        }
    }

    for(int y = 0; y < img.pad; y++){
        for(int x = -img.pad; x < img.width + img.pad; x++){
            IMG_AT(&img, -y-1, x) = IMG_AT(&img, y+1, x);
            IMG_AT(&img, img.height+y, x) = IMG_AT(&img, img.height-2-y, x);
        }
    }

    stbi_image_free(data);
    return img;
}

static float clampf(float v, float lo, float hi){
    if(v < lo) return lo;
    if(v > hi) return hi;
    return v;
}

void image_write(const image_t *img, const char *path){
    unsigned char *buf = malloc((size_t)img->width * (size_t)img->height);
    if(NULL == buf){
        fprintf(stderr, "image_write: out of memory\n");
        return;
    }

    for(int y = 0; y < img->height; y++){
        for(int x = 0; x < img->width; x++){
            float v = clampf(IMG_AT(img, y, x), 0.0f, 255.0f);
            buf[y * img->width + x] = (unsigned char)v;
        }
    }

    if(!stbi_write_png(path, img->width, img->height, 1, buf, img->width)){
        fprintf(stderr, "image_write: failed to write %s\n", path);
    }

    free(buf);
}

void image_free(image_t *img){
    free(img->data);
    img->data = NULL;
}