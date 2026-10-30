// Reference greedy decoding from llama.cpp's CPU backend for a tiny GGUF fixture (tests/contracts/test_arch.py compares
// ours). Same context settings as llama_logits.cpp (float32 K/V cache, flash attention off, 4 threads).
// (tools/gguf_fixtures/llama_greedy.cpp is the other greedy tool: real files and texts, for gguf_conform.py.)
// build at the pinned commit (tools/gguf_fixtures/greedy.py make does that):
//   g++ -O2 -std=c++17 llama_greedy_ids.cpp -I<llama.cpp>/include -I<llama.cpp>/ggml/include -L<build>/bin -lllama -o llama_greedy_ids
// usage: llama_greedy_ids MODEL.gguf OUT.f32 N_NEW TOKEN_ID...
//   one llama_decode of the prompt (logits of its last token), then N_NEW - 1 one-token decodes of the argmax (K/V cache
//   path). Writes N_NEW x n_vocab float32 (row s = the logits step s picks from) and prints one line per step:
//   "<step> <argmax id> <top-two gap>" (argmax = the lowest id among equal maxima; gap = top value - second value).
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s MODEL.gguf OUT.f32 N_NEW TOKEN_ID...\n", argv[0]);
        return 2;
    }
    const int n_new = atoi(argv[3]);
    std::vector<llama_token> toks;
    for (int i = 4; i < argc; i++) toks.push_back((llama_token) atoi(argv[i]));
    const int n = (int) toks.size();
    if (n_new < 1 || n < 1) { fprintf(stderr, "need N_NEW >= 1 and a prompt\n"); return 2; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + n_new + 16;
    cp.n_batch = n;
    cp.n_ubatch = n;
    cp.n_seq_max = 1;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * f = fopen(argv[2], "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == n - 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    for (int s = 0; s < n_new; s++) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        fwrite(lg, sizeof(float), nv, f);
        int best = 0;
        for (int v = 1; v < nv; v++) if (lg[v] > lg[best]) best = v;
        int second = best == 0 ? 1 : 0;
        for (int v = 0; v < nv; v++) if (v != best && lg[v] > lg[second]) second = v;
        printf("%d %d %.9g\n", s, best, (double) (lg[best] - lg[second]));
        if (s == n_new - 1) break;
        batch.n_tokens = 1;
        batch.token[0] = best;
        batch.pos[0] = n + s;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at step %d\n", s); return 1; }
    }
    fclose(f);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
