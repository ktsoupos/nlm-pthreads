/**
 * @brief Grayscale image stored as float pixels with a reflected border.
 *
 * @c data points at the top-left corner of the *padded* buffer, i.e. logical
 * coordinate (-pad, -pad), not at the first real pixel. Pixels are always
 * accessed through IMG_AT(), which applies the pad offset and lets callers
 * pass coordinates in [-pad, width/height + pad) with no bounds checks.
 *
 * The border of width @c pad is filled by reflect_101 (the edge pixel is not
 * duplicated: index -1 mirrors index 1, not index 0), so patch/search-window
 * reads never fall off the array even for pixels at the image edge.
 */
typedef struct {
    float *data;        /**< Padded buffer, allocated via aligned_alloc(64, ...). */
    int width, height;  /**< Logical (unpadded) image size, in pixels. */
    int pad;             /**< Border width: patch_radius + search_radius. */
    int stride;           /**< Floats per row: width + 2*pad, rounded up to a
                                multiple of 16 so rows stay usable for AVX2 loads. */
} image_t;


/**
 * @brief Access a pixel by logical coordinate, padding included.
 *
 * @c y and @c x may range over [-pad, height/width + pad) — negative or
 * past-the-edge coordinates land in the reflected border, not out of bounds.
 *
 * @param img  image_t* (not image_t) to index into.
 * @param y    Row, 0 is the first real row.
 * @param x    Column, 0 is the first real column.
 * @return     lvalue float for the pixel; usable on either side of an assignment.
 */
#define IMG_AT(img, y, x) \
    ((img)->data[((y) + (img)->pad) * (img)->stride + ((x) + (img)->pad)])


/**
 * @brief Load a grayscale PNG and return it as a padded float image.
 *
 * Forces single-channel decoding regardless of the source file's channel
 * count, converts uint8 -> float in [0,255], and fills a border of width
 * @p pad using reflect_101 padding.
 *
 * @param path  Path to a PNG file.
 * @param pad   Border width to allocate and fill (patch_radius + search_radius).
 * @return      image_t with @c data == NULL on failure; check before use and
 *              call stbi_failure_reason() for the cause.
 */
image_t image_load(const char *path, int pad);

/**
 * @brief Write the logical (unpadded) region of an image to a PNG file.
 *
 * Pixel values are clamped to [0,255] before the float -> uint8 cast; the
 * padded border is never written.
 *
 * @param img   Image to write; only [0,width) x [0,height) is used.
 * @param path  Destination path. PNG only.
 */
void    image_write(const image_t *img, const char *path);

/**
 * @brief Free an image's pixel buffer.
 *
 * @param img  Image previously returned by image_load(). Safe to call with
 *             @c data == NULL.
 */
void    image_free(image_t *img);
