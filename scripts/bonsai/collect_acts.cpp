// Collect last-token residual stream activations from the PrismML fork.
// Input: already chat-formatted prompts separated by 0x1e.
// Output: float32 [n_prompts, n_layer + 1, n_embd]; row 0 = embedding output
// (model.input_embed), row L = l_out of block L-1 (= input of block L).
//
//   collect_acts -m PTQ1_0.gguf -i prompts.rs -o acts.f32 [--lora A.gguf --scale S]

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct collector {
    int64_t n_embd = 0;
    int     n_rows = 0;
    std::vector<float> row;   // current prompt, [n_rows, n_embd]
    std::vector<char>  seen;
    std::vector<float> buf;
};

static int row_of(const char * name) {
    if (strcmp(name, "model.input_embed") == 0) return 0;
    if (strncmp(name, "l_out-", 6) == 0) return atoi(name + 6) + 1;
    return -1;
}

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    const int r = row_of(t->name);
    if (ask) return r >= 0;
    auto * c = (collector *) ud;
    if (r < 0 || r >= c->n_rows) return true;
    if (t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd || t->nb[0] != sizeof(float)) {
        fprintf(stderr, "unexpected %s: type %d ne0 %lld\n", t->name, t->type, (long long) t->ne[0]);
        exit(1);
    }
    c->buf.resize(ggml_nbytes(t) / sizeof(float) + 1);
    ggml_backend_tensor_get(t, c->buf.data(), 0, ggml_nbytes(t));
    const int64_t last = ggml_nrows(t) - 1;
    memcpy(c->row.data() + r * c->n_embd, c->buf.data() + last * (t->nb[1] / sizeof(float)), c->n_embd * sizeof(float));
    c->seen[r] = 1;
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, in_path, out_path, lora_path;
    float scale = 1.0f;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if      (a == "-m")      model_path = argv[i + 1];
        else if (a == "-i")      in_path    = argv[i + 1];
        else if (a == "-o")      out_path   = argv[i + 1];
        else if (a == "--lora")  lora_path  = argv[i + 1];
        else if (a == "--scale") scale      = std::stof(argv[i + 1]);
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || in_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -i prompts.rs -o acts.f32 [--lora a.gguf] [--scale s]\n", argv[0]);
        return 1;
    }

    std::vector<std::string> prompts;
    {
        std::ifstream f(in_path, std::ios::binary);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string all = ss.str(), cur;
        for (char ch : all) {
            if (ch == '\x1e') { if (!cur.empty()) prompts.push_back(cur); cur.clear(); }
            else cur += ch;
        }
        if (!cur.empty()) prompts.push_back(cur);
    }

    llama_log_set([](ggml_log_level lvl, const char * msg, void *) {
        if (lvl >= GGML_LOG_LEVEL_WARN) fputs(msg, stderr);
    }, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    collector c;
    c.n_embd = llama_model_n_embd(model);
    c.n_rows = llama_model_n_layer(model) + 1;
    c.row.assign(c.n_rows * c.n_embd, 0.0f);
    c.seen.assign(c.n_rows, 0);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = 2048;
    cp.n_batch           = 2048;
    cp.n_ubatch          = 2048;
    cp.cb_eval           = cb_eval;
    cp.cb_eval_user_data = &c;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }

    if (!lora_path.empty()) {
        llama_adapter_lora * ad = llama_adapter_lora_init(model, lora_path.c_str());
        if (!ad) { fprintf(stderr, "lora load failed\n"); return 1; }
        llama_set_adapters_lora(ctx, &ad, 1, &scale);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    FILE * out = fopen(out_path.c_str(), "wb");
    if (!out) { fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 1; }

    for (size_t i = 0; i < prompts.size(); ++i) {
        const std::string & p = prompts[i];
        std::vector<llama_token> toks(p.size() + 8);
        // template text already carries every special token
        const int n = llama_tokenize(vocab, p.c_str(), (int) p.size(), toks.data(), (int) toks.size(), false, true);
        if (n <= 0 || n > (int) cp.n_ctx) { fprintf(stderr, "prompt %zu: bad token count %d\n", i, n); return 1; }
        toks.resize(n);

        llama_memory_clear(llama_get_memory(ctx), true);
        std::fill(c.seen.begin(), c.seen.end(), 0);
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), n)) != 0) {
            fprintf(stderr, "prompt %zu: decode failed\n", i);
            return 1;
        }
        for (int r = 0; r < c.n_rows; ++r) {
            if (!c.seen[r]) { fprintf(stderr, "prompt %zu: row %d not captured\n", i, r); return 1; }
        }
        fwrite(c.row.data(), sizeof(float), c.row.size(), out);
        if ((i + 1) % 50 == 0 || i + 1 == prompts.size()) {
            fprintf(stderr, "[collect] %zu/%zu (last prompt %d tokens)\n", i + 1, prompts.size(), n);
        }
    }
    fclose(out);
    printf("wrote %s: [%zu, %d, %lld] float32\n", out_path.c_str(), prompts.size(), c.n_rows, (long long) c.n_embd);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
