// ggml's own quantize and dequantize for one block type (the reference side of tools/gguf_conform.py --quant), modeled
// on llama.cpp's tests/test-quantize-fns.cpp. Uses only ggml's public C API (ggml/include/ggml.h).
//   cc -O2 ggml_dequant.c -I<llama.cpp>/ggml/include -L<build>/bin -lggml -lggml-base -o ggml_dequant
// usage: ggml_dequant TYPE_ID ROWS COLS SEED OUT_PREFIX
//   Draws ROWS x COLS floats (a fixed LCG from SEED: uniform in [-1, 1) times a per-row scale in [0.01, 10), so blocks
//   meet small and large scales), quantizes them with ggml_quantize_chunk (a uniform importance matrix of ones for the
//   types that require one), writes OUT_PREFIX.q (the blocks as stored in a GGUF file) and OUT_PREFIX.f32 (ggml's
//   to_float of those blocks, ROWS x COLS float32). Exit 3 when ggml cannot quantize the type.
#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static uint64_t s;
static float u01(void) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return (float) ((s >> 40) & 0xFFFFFF) / 16777216.0f; }

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: ggml_dequant TYPE_ID ROWS COLS SEED OUT_PREFIX\n"); return 2; }
    enum ggml_type t = (enum ggml_type) atoi(argv[1]);
    int64_t rows = atoll(argv[2]), cols = atoll(argv[3]);
    s = (uint64_t) atoll(argv[4]);
    ggml_cpu_init();
    const struct ggml_type_traits * tr = ggml_get_type_traits(t);
    if (!tr || !tr->to_float || tr->blck_size == 0 || cols % tr->blck_size) { fprintf(stderr, "type %d: no to_float or bad cols\n", t); return 3; }
    float * x = malloc(sizeof(float) * rows * cols), * y = malloc(sizeof(float) * rows * cols), * im = malloc(sizeof(float) * cols);
    for (int64_t r = 0; r < rows; r++) {
        float sc = 0.01f + 10.0f * u01() * u01();
        for (int64_t c = 0; c < cols; c++) x[r * cols + c] = (2.0f * u01() - 1.0f) * sc;
    }
    for (int64_t c = 0; c < cols; c++) im[c] = 1.0f;
    size_t rb = ggml_row_size(t, cols);
    void * q = malloc(rb * rows);
    ggml_quantize_init(t);
    size_t n = ggml_quantize_chunk(t, x, q, 0, rows, cols, ggml_quantize_requires_imatrix(t) ? im : NULL);
    if (n != rb * rows) { fprintf(stderr, "type %d: quantize wrote %zu of %zu bytes\n", t, n, rb * rows); return 3; }
    for (int64_t r = 0; r < rows; r++) tr->to_float((const char *) q + r * rb, y + r * cols, cols);
    char p[4096];
    snprintf(p, sizeof p, "%s.q", argv[5]);
    FILE * f = fopen(p, "wb"); fwrite(q, 1, rb * rows, f); fclose(f);
    snprintf(p, sizeof p, "%s.f32", argv[5]);
    f = fopen(p, "wb"); fwrite(y, sizeof(float), rows * cols, f); fclose(f);
    printf("%s %zu\n", tr->type_name, rb);
    return 0;
}
