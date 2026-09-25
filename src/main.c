#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image.h"
#include "nlm.h"
#include "timer.h"

static const struct { const char *name; nlm_variant v; } VARIANTS[] = {
    { "seq",            NLM_SEQ               },
    { "seq_simd",       NLM_SEQ_SIMD          },
    { "pthreads",       NLM_PTHREADS          },
    { "pthreads_simd",  NLM_PTHREADS_SIMD     },
    { "integral",       NLM_INTEGRAL          },
    { "integral_thr",   NLM_INTEGRAL_PTHREADS },
};
static const int NVARIANTS = (int)(sizeof VARIANTS / sizeof VARIANTS[0]);

static void usage(const char *prog){
    fprintf(stderr,
        "usage: %s --in noisy.png [options]\n"
        "  --out PATH        write denoised image\n"
        "  --clean PATH      clean reference, enables PSNR\n"
        "  --variant NAME    seq|seq_simd|pthreads|pthreads_simd|integral|integral_thr\n"
        "  --threads N       worker threads (default 1)\n"
        "  --patch R         patch radius (default 3)\n"
        "  --search R        search radius (default 10)\n"
        "  --sigma S         noise sigma (default 25)\n"
        "  --h H             filter strength (default 0.4*sigma)\n"
        "  --repeat N        repeat the timed region N times, one CSV row each\n"
        "  --csv PATH        append results as CSV\n"
        "  --check           also run seq and report max|diff| against it\n", prog);
}

static double psnr_vs(const image_t *img, const char *clean_path){
    int w, h, n;
    unsigned char *ref = stbi_load(clean_path, &w, &h, &n, 1);
    if(!ref || w != img->width || h != img->height){
        if(ref) stbi_image_free(ref);
        return NAN;
    }
    double mse = 0.0;
    for(int y = 0; y < h; y++){
        for(int x = 0; x < w; x++){
            float v = IMG_AT(img, y, x);
            if(v < 0.0f) v = 0.0f;
            if(v > 255.0f) v = 255.0f;
            double d = (double)ref[y*w + x] - (double)(unsigned char)v;
            mse += d * d;
        }
    }
    stbi_image_free(ref);
    mse /= (double)w * h;
    return mse > 0.0 ? 10.0 * log10(255.0*255.0 / mse) : INFINITY;
}

int main(int argc, char **argv){
    const char *in_path = NULL, *out_path = NULL, *clean_path = NULL, *csv_path = NULL;
    const char *variant_name = "seq";
    int threads = 1, repeat = 1, check = 0;
    float sigma = 25.0f, h = -1.0f;
    nlm_params p = { .patch_radius = 3, .search_radius = 10 };

    for(int i = 1; i < argc; i++){
        const char *a = argv[i];
        #define NEXT() (++i < argc ? argv[i] : (usage(argv[0]), exit(1), ""))
        if     (!strcmp(a,"--in"))      in_path      = NEXT();
        else if(!strcmp(a,"--out"))     out_path     = NEXT();
        else if(!strcmp(a,"--clean"))   clean_path   = NEXT();
        else if(!strcmp(a,"--csv"))     csv_path     = NEXT();
        else if(!strcmp(a,"--variant")) variant_name = NEXT();
        else if(!strcmp(a,"--threads")) threads      = atoi(NEXT());
        else if(!strcmp(a,"--patch"))   p.patch_radius  = atoi(NEXT());
        else if(!strcmp(a,"--search"))  p.search_radius = atoi(NEXT());
        else if(!strcmp(a,"--sigma"))   sigma        = (float)atof(NEXT());
        else if(!strcmp(a,"--h"))       h            = (float)atof(NEXT());
        else if(!strcmp(a,"--repeat"))  repeat       = atoi(NEXT());
        else if(!strcmp(a,"--check"))   check        = 1;
        else { usage(argv[0]); return 1; }
        #undef NEXT
    }
    if(!in_path){ usage(argv[0]); return 1; }

    p.sigma   = sigma;
    p.h       = (h > 0.0f) ? h : 0.4f * sigma;   // CLAUDE.md's rule of thumb
    p.threads = threads;

    nlm_variant variant = NLM_SEQ;
    int found = 0;
    for(int i = 0; i < NVARIANTS; i++)
        if(!strcmp(variant_name, VARIANTS[i].name)){ variant = VARIANTS[i].v; found = 1; break; }
    if(!found){ fprintf(stderr, "unknown variant '%s'\n", variant_name); return 1; }

    const int pad = p.patch_radius + p.search_radius;
    image_t in  = image_load(in_path, pad);
    image_t out = image_load(in_path, pad);
    if(!in.data || !out.data){
        fprintf(stderr, "load failed: %s\n", stbi_failure_reason());
        return 1;
    }

    double max_abs_diff = NAN;
    if(check){
        image_t ref = image_load(in_path, pad);
        nlm_run(&in, &ref, &p, NLM_SEQ);
        nlm_run(&in, &out, &p, variant);
        max_abs_diff = 0.0;
        for(int y = 0; y < in.height; y++)
            for(int x = 0; x < in.width; x++){
                double d = fabs((double)IMG_AT(&ref,y,x) - (double)IMG_AT(&out,y,x));
                if(d > max_abs_diff) max_abs_diff = d;
            }
        image_free(&ref);
    }

    FILE *csv = NULL;
    if(csv_path){
        int fresh = 1;
        FILE *probe = fopen(csv_path, "r");
        if(probe){ fresh = (fgetc(probe) == EOF); fclose(probe); }
        csv = fopen(csv_path, "a");
        if(!csv){ fprintf(stderr, "cannot open %s\n", csv_path); return 1; }
        if(fresh) fprintf(csv, "variant,threads,patch_radius,search_radius,sigma,h,"
                               "width,height,run,seconds,psnr,max_abs_diff\n");
    }

    for(int r = 0; r < repeat; r++){
        stopwatch_t sw;
        timer_start(&sw);
        nlm_run(&in, &out, &p, variant);            // timed region: compute only
        double secs = timer_elapsed_sec(&sw);

        double q = clean_path ? psnr_vs(&out, clean_path) : NAN;

        printf("%-14s threads=%-3d patch=%d search=%d sigma=%.1f h=%.2f  %8.4f s",
               variant_name, p.threads, p.patch_radius, p.search_radius, p.sigma, p.h, secs);
        if(clean_path)        printf("  PSNR %.2f dB", q);
        if(!isnan(max_abs_diff)) printf("  max|diff| %g", max_abs_diff);
        printf("\n");

        if(csv){
            fprintf(csv, "%s,%d,%d,%d,%.4f,%.4f,%d,%d,%d,%.6f,",
                    variant_name, p.threads, p.patch_radius, p.search_radius,
                    p.sigma, p.h, in.width, in.height, r, secs);
            if(isnan(q)) fprintf(csv, ","); else fprintf(csv, "%.4f,", q);
            if(isnan(max_abs_diff)) fprintf(csv, "\n"); else fprintf(csv, "%g\n", max_abs_diff);
        }
    }

    if(csv) fclose(csv);
    if(out_path) image_write(&out, out_path);

    image_free(&in);
    image_free(&out);
    return 0;
}
