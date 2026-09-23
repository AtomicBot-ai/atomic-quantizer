// Runtime check of a refusal-ablation LoRA inside the PrismML fork.
// Taps every residual write (embd, linear_attn_out, attn_output, ffn_out) and the
// residual stream itself (l_out) with cb_eval, and prints leak = |r.y| / |y|.
//
//   leak_probe -m PTQ1_0.gguf -d refusal_dir_fp32.bin [--lora A.gguf [--scale S]] [-p TEXT]

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct stat_acc {
    double sum = 0.0;
    double max = 0.0;
    int    n   = 0;
};

struct probe {
    std::vector<float> r;
    std::vector<float> buf;
    std::map<std::string, stat_acc> fam;   // family -> stats over layers and tokens
    std::map<int, stat_acc>         lout;  // layer -> residual stream stats
};

static const char * TAPS[] = { "embd", "linear_attn_out-", "attn_output-", "ffn_out-", "l_out-" };

static bool want(const char * name) {
    for (const char * p : TAPS) {
        const size_t n = strlen(p);
        if (p[n - 1] == '-' ? strncmp(name, p, n) == 0 : strcmp(name, p) == 0) {
            return true;
        }
    }
    return false;
}

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    if (ask) {
        return want(t->name);
    }
    if (!want(t->name)) {
        return true;
    }
    auto * p = (probe *) ud;
    const int64_t n = t->ne[0];
    if (t->type != GGML_TYPE_F32 || n != (int64_t) p->r.size() || t->nb[0] != sizeof(float)) {
        fprintf(stderr, "skip %s: type %d ne0 %lld\n", t->name, t->type, (long long) n);
        return true;
    }
    p->buf.resize(ggml_nbytes(t) / sizeof(float) + 1);
    ggml_backend_tensor_get(t, p->buf.data(), 0, ggml_nbytes(t));

    const std::string name = t->name;
    const size_t dash = name.rfind('-');
    const std::string fam = dash == std::string::npos ? name : name.substr(0, dash);
    const int layer = dash == std::string::npos ? -1 : atoi(name.c_str() + dash + 1);

    const int64_t n_tok = ggml_nrows(t);
    for (int64_t i = 0; i < n_tok; ++i) {
        const float * y = p->buf.data() + i * (t->nb[1] / sizeof(float));
        double dot = 0.0, nrm = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            dot += (double) y[j] * p->r[j];
            nrm += (double) y[j] * y[j];
        }
        const double leak = std::fabs(dot) / std::sqrt(nrm + 1e-30);
        stat_acc & s = fam == "l_out" ? p->lout[layer] : p->fam[fam];
        s.sum += leak;
        s.max  = std::max(s.max, leak);
        s.n   += 1;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, dir_path, lora_path;
    std::string text = "The history of the printing press begins in the fifteenth century, when Johannes "
                       "Gutenberg combined movable metal type, oil-based ink and a wooden screw press. "
                       "Within fifty years presses operated in more than two hundred European cities.";
    float scale = 1.0f;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if      (a == "-m")      model_path = argv[i + 1];
        else if (a == "-d")      dir_path   = argv[i + 1];
        else if (a == "--lora")  lora_path  = argv[i + 1];
        else if (a == "--scale") scale      = std::stof(argv[i + 1]);
        else if (a == "-p")      text       = argv[i + 1];
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || dir_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -d dir_fp32.bin [--lora a.gguf] [--scale s] [-p text]\n", argv[0]);
        return 1;
    }

    probe p;
    {
        std::ifstream f(dir_path, std::ios::binary | std::ios::ate);
        const size_t n = f.tellg() / sizeof(float);
        p.r.resize(n);
        f.seekg(0);
        f.read((char *) p.r.data(), n * sizeof(float));
        double nrm = 0.0;
        for (float v : p.r) nrm += (double) v * v;
        for (float & v : p.r) v = (float) (v / std::sqrt(nrm));
    }

    llama_log_set([](ggml_log_level lvl, const char * msg, void *) {
        if (lvl >= GGML_LOG_LEVEL_WARN) fputs(msg, stderr);
    }, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = 1024;
    cp.n_batch           = 1024;
    cp.n_ubatch          = 1024;
    cp.cb_eval           = cb_eval;
    cp.cb_eval_user_data = &p;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }

    if (!lora_path.empty()) {
        llama_adapter_lora * ad = llama_adapter_lora_init(model, lora_path.c_str());
        if (!ad) { fprintf(stderr, "lora load failed\n"); return 1; }
        llama_set_adapters_lora(ctx, &ad, 1, &scale);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(text.size() + 8);
    const int n_tok = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    toks.resize(n_tok);
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n_tok)) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    printf("adapter=%s scale=%g tokens=%d\n", lora_path.empty() ? "none" : lora_path.c_str(), scale, n_tok);
    printf("  %-18s %6s %11s %11s\n", "writer", "n", "mean_leak", "max_leak");
    for (const auto & [k, s] : p.fam) {
        printf("  %-18s %6d %11.3e %11.3e\n", k.c_str(), s.n, s.sum / s.n, s.max);
    }
    double lmax = 0.0;
    printf("  residual l_out mean leak by layer:");
    for (const auto & [il, s] : p.lout) {
        lmax = std::max(lmax, s.max);
        if (il % 8 == 0 || il == (int) p.lout.size() - 1) printf(" L%d=%.2e", il, s.sum / s.n);
    }
    printf("\n  residual l_out max leak over all layers: %.3e\n", lmax);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
