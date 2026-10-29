// Token ids from llama.cpp's own tokenizer, as `llama-tokenize --ids --no-bos`: llama_tokenize(vocab, text,
// add_special = false, parse_special = true) on a vocabulary-only load of the file.
// Usage: llama_tokenize MODEL.gguf TEXTS   (TEXTS: the texts, each followed by one NUL byte). One line of ids per text.
#include "llama.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void quiet(enum ggml_log_level, const char *, void *) {}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf TEXTS\n", argv[0]);
        return 2;
    }
    llama_log_set(quiet, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::ifstream in(argv[2], std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string all = ss.str();
    size_t start = 0;
    while (start < all.size()) {
        size_t end = all.find('\0', start);
        if (end == std::string::npos) {
            end = all.size();
        }
        const std::string text = all.substr(start, end - start);
        std::vector<llama_token> ids(text.size() + 16);
        int n = llama_tokenize(vocab, text.data(), (int) text.size(), ids.data(), (int) ids.size(), false, true);
        if (n < 0) {
            ids.resize(-n);
            n = llama_tokenize(vocab, text.data(), (int) text.size(), ids.data(), (int) ids.size(), false, true);
        }
        printf("[");
        for (int i = 0; i < n; i++) {
            printf(i ? ", %d" : "%d", ids[i]);
        }
        printf("]\n");
        start = end + 1;
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
