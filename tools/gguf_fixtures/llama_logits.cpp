// Reference logits from llama.cpp's CPU backend for a tiny GGUF fixture (tests/cpu/test_gguf_<arch>.py compares ours).
// Uses only llama.cpp's public C API (include/llama.h); it is a test tool, not a runner. Build it against a llama.cpp
// build at the pinned commit (tools/gguf_fixtures.py logits does that):
//   g++ -O2 -std=c++17 llama_logits.cpp -I<llama.cpp>/include -I<llama.cpp>/ggml/include -L<build>/bin -lllama -o llama_logits
// usage: llama_logits MODEL.gguf OUT.f32 TOKEN_ID...   (writes n_tokens x n_vocab float32, row-major)
//        llama_logits --tokenize MODEL.gguf TEXT_FILE   (prints the token ids of the file's text, no special tokens)
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int tokenize(const char * model_path, const char * text_path) {
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    FILE * f = fopen(text_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", text_path); return 1; }
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
    fclose(f);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(text.size() + 16);
    int k = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true);
    if (k < 0) { toks.resize(-k); k = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true); }
    for (int i = 0; i < k; i++) printf("%d\n", toks[i]);
    llama_model_free(model);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc >= 4 && strcmp(argv[1], "--tokenize") == 0) {
        llama_backend_init();
        int r = tokenize(argv[2], argv[3]);
        llama_backend_free();
        return r;
    }
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf OUT.f32 TOKEN_ID...\n", argv[0]);
        return 2;
    }
    std::vector<llama_token> toks;
    for (int i = 3; i < argc; i++) toks.push_back((llama_token) atoi(argv[i]));
    const int n = (int) toks.size();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + 16;
    cp.n_batch = n;
    cp.n_ubatch = n;
    cp.n_seq_max = 1;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.type_k = GGML_TYPE_F32;     // float32 K/V: the comparison is about the graph, not cache rounding
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * f = fopen(argv[2], "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    for (int i = 0; i < n; i++) fwrite(llama_get_logits_ith(ctx, i), sizeof(float), nv, f);
    fclose(f);
    printf("%d %d\n", n, nv);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
