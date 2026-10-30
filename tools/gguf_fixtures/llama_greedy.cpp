// llama.cpp's own greedy continuation and logits for a real GGUF file, CPU backend (the reference side of
// tools/gguf_conform.py). Uses only llama.cpp's public C API (include/llama.h); a test tool, not a runner.
//   g++ -O2 -std=c++17 llama_greedy.cpp -I<llama.cpp>/include -I<llama.cpp>/ggml/include -L<build>/bin -lllama -o llama_greedy
// usage: llama_greedy MODEL.gguf N_GEN OUT_PREFIX TEXT_FILE...
//   For each text file k: tokenizes it as llama.cpp does for a prompt (BOS per the vocabulary's add_bos, special tokens
//   parsed), decodes the prompt in one batch with logits at every position, then N_GEN greedy steps (argmax, one token
//   per decode). Writes OUT_PREFIX<k>.ids (prompt ids, a blank line, generated ids) and OUT_PREFIX<k>.f32: the logits
//   that predicted each next token, (n_prompt + N_GEN - 1) x n_vocab float32, row-major.
#include "llama.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: llama_greedy MODEL N_GEN OUT_PREFIX TEXT_FILE...\n"); return 2; }
    const int n_gen = atoi(argv[2]);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int k = 4; k < argc; k++) {
        FILE * f = fopen(argv[k], "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[k]); return 1; }
        std::string text; char buf[4096]; size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
        fclose(f);
        std::vector<llama_token> toks(text.size() + 16);
        int nt = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, true);
        if (nt < 0) { toks.resize(-nt); nt = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, true); }
        toks.resize(nt);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = nt + n_gen + 8; cp.n_batch = cp.n_ctx; cp.n_ubatch = cp.n_ctx; cp.no_perf = true;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
        std::vector<float> out;
        llama_batch b = llama_batch_init(nt, 0, 1);
        for (int i = 0; i < nt; i++) {
            b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        b.n_tokens = nt;
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
        for (int i = 0; i < nt; i++) { const float * l = llama_get_logits_ith(ctx, i); out.insert(out.end(), l, l + n_vocab); }
        llama_batch_free(b);
        std::vector<llama_token> gen;
        const float * last = out.data() + (size_t) (nt - 1) * n_vocab;
        for (int s = 0; s < n_gen; s++) {
            int best = 0;
            for (int v = 1; v < n_vocab; v++) if (last[v] > last[best]) best = v;
            gen.push_back(best);
            if (s == n_gen - 1) break;
            llama_batch one = llama_batch_init(1, 0, 1);
            one.token[0] = best; one.pos[0] = nt + s; one.n_seq_id[0] = 1; one.seq_id[0][0] = 0; one.logits[0] = 1; one.n_tokens = 1;
            if (llama_decode(ctx, one)) { fprintf(stderr, "decode failed\n"); return 1; }
            const float * l = llama_get_logits_ith(ctx, 0);
            out.insert(out.end(), l, l + n_vocab);
            last = out.data() + out.size() - n_vocab;
            llama_batch_free(one);
        }
        std::string pre = std::string(argv[3]) + std::to_string(k - 4);
        FILE * fi = fopen((pre + ".ids").c_str(), "w");
        for (auto t : toks) fprintf(fi, "%d ", t);
        fprintf(fi, "\n\n");
        for (auto t : gen) fprintf(fi, "%d ", t);
        fprintf(fi, "\n");
        fclose(fi);
        FILE * fo = fopen((pre + ".f32").c_str(), "wb");
        fwrite(out.data(), sizeof(float), out.size(), fo);
        fclose(fo);
        llama_free(ctx);
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
