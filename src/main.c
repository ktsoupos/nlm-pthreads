#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image.h"
#include "nlm.h"
#include "timer.h"

int main(int argc, char **argv){
    if (argc < 2) {
        fprintf(stderr, "usage: %s <noisy_in.png> [out.png]\n", argv[0]);
        return 1;
    }
    const char *in_path  = argv[1];
    const char *out_path = (argc >= 3) ? argv[2] : "data/out/denoised.png";

    nlm_params p = {
        .patch_radius  = 3,
        .search_radius = 10,
        .sigma         = 25.0f,
        .h             = 10.0f,
        .threads       = 1,
    };
    int pad = p.patch_radius + p.search_radius;

    image_t img = image_load(in_path, pad);
    if (!img.data) {
        fprintf(stderr, "load failed: %s\n", stbi_failure_reason());
        return 1;
    }
    printf("%dx%d, pad=%d, stride=%d\n", img.width, img.height, img.pad, img.stride);

    // scratch buffer, same size/padding as img; contents get fully overwritten by nlm_run
    image_t out = image_load(in_path, pad);
    if (!out.data) {
        fprintf(stderr, "alloc failed\n");
        image_free(&img);
        return 1;
    }

    stopwatch_t sw;
    timer_start(&sw);
    nlm_run(&img, &out, &p, NLM_SEQ);
    double elapsed = timer_elapsed_sec(&sw);
    printf("nlm_seq: %.4f s\n", elapsed);

    image_write(&out, out_path);
    printf("wrote %s\n", out_path);

    image_free(&img);
    image_free(&out);
    return 0;
}