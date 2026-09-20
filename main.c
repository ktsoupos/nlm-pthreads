#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb/stb_image_write.h"

int main(void){
    int w, h, n;
    unsigned char *data = stbi_load("test/images/pengbrew.png", &w, &h, &n, 0);
    if (!data) {
        fprintf(stderr, "load failed: %s\n", stbi_failure_reason());
        return 1;
    }
    printf("%dx%d, %d channels\n", w, h, n);

    stbi_image_free(data);
    return 0;
}