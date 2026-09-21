#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image.h"

int main(void){
    const int patch_radius = 3;
    const int search_radius = 10;

    image_t img = image_load("test/images/pengbrew.png", patch_radius + search_radius);
    if (!img.data) {
        fprintf(stderr, "load failed: %s\n", stbi_failure_reason());
        return 1;
    }
    printf("%dx%d, pad=%d, stride=%d\n", img.width, img.height, img.pad, img.stride);

    image_free(&img);
    return 0;
}