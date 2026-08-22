// oracle.c — single-shot llama.cpp oracle: tokenizes a prompt, generates N
// tokens with pure argmax sampling (matches pesti coherence_check's temp=0.0),
// and prints prompt token IDs + generated token IDs + decoded text.
//
// Build:
//   gcc -O2 oracle.c -I/home/crombo/llama.cpp/include \
//     -L/home/crombo/llama.cpp/build-cpu/bin -lllama -lggml -lggml-base -lggml-cpu \
//     -Wl,-rpath,/home/crombo/llama.cpp/build-cpu/bin -o oracle
//
// Usage: oracle <model.gguf> <max_tokens> <prompt...>

#include <llama.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_TOKENS 8192

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    int use_special = 0;
    int argi = 1;
    if (argc >= 2 && strcmp(argv[1], "--special") == 0) { use_special = 1; argi = 2; }
    if (argc < argi + 3) {
        fprintf(stderr, "usage: %s [--special] <model.gguf> <max_tokens> <prompt...|@file>\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[argi];
    int max_tokens = atoi(argv[argi + 1]);
    // prompt: either joined remaining args, or @file
    char prompt[8192] = {0};
    int plen = 0;
    const char *first = argv[argi + 2];
    if (first[0] == '@') {
        FILE *f = fopen(first + 1, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", first + 1); return 2; }
        size_t rd = fread(prompt, 1, sizeof(prompt) - 1, f);
        fclose(f);
        plen = (int)rd;
        prompt[plen] = 0;
    } else {
        for (int i = argi + 2; i < argc; i++) {
            if (i > argi + 2) prompt[plen++] = ' ';
            size_t n = strlen(argv[i]);
            if (plen + (int)n + 1 > (int)sizeof(prompt)) {
                fprintf(stderr, "prompt too long\n");
                return 2;
            }
            memcpy(prompt + plen, argv[i], n);
            plen += (int)n;
        }
        prompt[plen] = 0;
    }

    const double t0 = now_sec();
    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "FATAL: failed to load model\n"); return 1; }
    const double t1 = now_sec();

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // tokenize prompt (no special tokens — matches pesti's plain encode)
    llama_token prompt_tokens[MAX_TOKENS];
    int32_t n_prompt_raw = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
                                      prompt_tokens, MAX_TOKENS, /*add_special=*/false, /*parse_special=*/use_special != 0);
    if (n_prompt_raw < 0) { fprintf(stderr, "FATAL: tokenize failed (err=%d)\n", n_prompt_raw); return 1; }
    int32_t n_prompt = n_prompt_raw;

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_threads = 20;
    cparams.n_threads_batch = 20;
    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "FATAL: failed to init context\n"); return 1; }

    printf("[load] %.2fs\n", t1 - t0);
    printf("[prompt] %d tokens: [", n_prompt);
    for (int32_t i = 0; i < n_prompt; i++) printf("%d%s", prompt_tokens[i], i + 1 < n_prompt ? ", " : "");
    printf("]\n");

    // encode prompt: logits only for the last position
    struct llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    batch.n_tokens = n_prompt;
    for (int32_t i = 0; i < n_prompt; i++) {
        batch.token[i] = prompt_tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_prompt - 1) ? 1 : 0;
    }
    int32_t r = llama_decode(ctx, batch);
    if (r != 0) { fprintf(stderr, "FATAL: prompt decode failed r=%d\n", r); return 1; }
    llama_batch_free(batch);

    int32_t cur_pos = n_prompt;
    int32_t cur_token = 0;
    llama_token gen_tokens[MAX_TOKENS];
    int32_t n_gen = 0;

    const double t2 = now_sec();
    for (int32_t step = 0; step < max_tokens; step++) {
        const float *logits = llama_get_logits(ctx);
        // argmax
        int32_t best = 0;
        float best_v = -1e30f;
        for (int32_t i = 0; i < n_vocab; i++) {
            if (logits[i] > best_v) { best_v = logits[i]; best = i; }
        }
        cur_token = best;
        gen_tokens[n_gen++] = cur_token;

        struct llama_batch b1 = llama_batch_init(1, 0, 1);
        b1.n_tokens = 1;
        b1.token[0] = cur_token;
        b1.pos[0] = cur_pos;
        b1.n_seq_id[0] = 1;
        b1.seq_id[0][0] = 0;
        b1.logits[0] = 1;
        if (llama_decode(ctx, b1) != 0) { fprintf(stderr, "FATAL: token decode failed at step %d\n", step); return 1; }
        llama_batch_free(b1);
        cur_pos++;
    }
    const double t3 = now_sec();

    printf("=== %d tokens in %.3fs (%.2f tok/s) ===\n", n_gen, t3 - t2, (double)n_gen / (t3 - t2));

    // decode text
    char text[65536];
    int32_t ntext = llama_detokenize(vocab, gen_tokens, n_gen, text, (int32_t)sizeof(text) - 1, 0, 1);
    if (ntext < 0) ntext = 0;
    text[ntext] = 0;
    printf("TEXT: %.300s\n", text);

    printf("GEN_TOKEN_IDS: [");
    for (int32_t i = 0; i < n_gen; i++) printf("%d%s", gen_tokens[i], i + 1 < n_gen ? ", " : "");
    printf("]\n");

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
