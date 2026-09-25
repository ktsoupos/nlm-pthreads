#ifndef IMAGE_H
#define IMAGE_H

/*
 * Grayscale float image with a reflect_101 border of width `pad` (the edge pixel
 * is not duplicated: index -1 mirrors index 1). `data` points at the top-left of
 * the *padded* buffer, i.e. logical (-pad, -pad), so all access goes through
 * IMG_AT. The border lets patch and search-window reads run past the image edge
 * without bounds checks.
 */
typedef struct {
    float *data;
    int width, height;  /* logical, unpadded */
    int pad;
    int stride;         /* width + 2*pad, rounded up to a multiple of 16 floats */
} image_t;

/* y and x may range over [-pad, height/width + pad). Yields an lvalue. */
#define IMG_AT(img, y, x) \
    ((img)->data[((y) + (img)->pad) * (img)->stride + ((x) + (img)->pad)])

/* Loads as grayscale regardless of source format. Returns data == NULL on
   failure; call stbi_failure_reason() for the cause. */
image_t image_load(const char *path, int pad);

/* Writes the logical region only, clamped to [0,255]. PNG. */
void    image_write(const image_t *img, const char *path);

void    image_free(image_t *img);

#endif
