# Slow-Friend Substrate: Bounded Memory, Scoped MoE, Drift-Gated Compaction

**Date**: 2026-09-02
**Status**: 📋 Concept / Direction — see **EDR-011** (`CHANGELOG.md`) and `ROADMAP.md` → "Phase 5"
**Companion to**: [`Computational Inertia Concept.md`](Computational%20Inertia%20Concept.md)

---

## Sources (reference backing)

| Ref | What it is |
|-----|-----------|
| **Qwen3.8-Flash-Next paper** — `arXiv-2608.30320v1` (tarball in this dir) | The reference architecture: GDN + QSA + N-gram + Gated Residual + MoE |
| `sections/gdn_hybrid.tex` | Gated DeltaNet recurrence, decay/write gates, 3-of-4-layer hybrid |
| `sections/qsa.tex` | Qwen Sparse Attention + MQA indexer (budget-bounded retrieval) |
| `sections/ngram.tex` | 51B deterministic N-gram embedding table, host offload + async prefetch |
| `sections/residual.tex` | Gated Residual, bounded positive gates, FP8 residual storage |
| HF model card | https://huggingface.co/Qwen/Qwen3.8-Flash-Next |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next |
| `ROADMAP.md` Week 17 | PESTI's *measured* GPU-vs-oracle drift (f16 tensor-core accumulation) |
| `Computational Inertia Concept.md` | The flywheel / "preserve intent, bound debt" principle this doc operationalizes |

---

## 0. One paragraph

Qwen3.8-Flash-Next already implements the slow/wild friend split as a *layer schedule*: three cheap recurrent layers (Gated DeltaNet) that compress history into a fixed-size state, one precise-retrieval layer (Qwen Sparse Attention), repeating — plus a 51B deterministic N-gram table that lives in host RAM. The GDN state is the **slow friend** (bounded, O(1), stable, CPU-cheap); sparse attention is the **wild friend** (precise but expensive and drift-prone at high context). This doc turns that split into a PESTI substrate principle: keep a cheap always-on stable reference on CPU; use it to (a) preserve momentum when the GPU drops, (b) bound and edit the fast path's output, (c) scope MoE/compute routing, and (d) trigger *drift-gated compaction* — re-anchoring both paths toward the stable summary when they diverge. The unifying rule: **bounded positive gates on everything that accumulates state; hard sparsity only for compute.**

---

## The Polarity: fast × slow are complementary, not ranked

This is a **polarity**, not a hierarchy. Neither friend is "smarter" or "lesser" than the other — each is *specialized* for one axis, and the system needs both axes present at once. Calling either a "moron" would be a category error: fast and slow here are properties of **variance vs stability** (reach vs grounding), not of capability. A good fast friend and a good slow friend are each doing their job *well*; the problem only appears when one pole is missing.

**The two failure modes are symmetric, and both are real:**

- **Fast without slow → ungrounded velocity.** The fast friend reaches far, moves quickly, and is often right — but with no stable reference it *cannot tell when it has drifted*. At high context its instantaneous state softens (lost-in-the-middle, FP8 accumulation), so a free run wanders and the error compounds. It gets you to places you were never ready for, and there's no place to come back from. Speed without a compass is just distance.
- **Slow without fast → bounded stasis.** The slow friend is cheap, stable, always-on — but it is a *low-pass filter*. Alone it smooths away exactly the sharp detail that makes precision possible; it never takes the reach, so it never gets you anywhere new. Stability without legs is just standing still.

Both failure modes come from **missing one pole**, not from either pole being weak. That's what "a polarity that needs to exist" means: the value lives in the *ratio* between them, and the ratio has to be tuned per context — which is exactly what the bounded confidence weight `w` (§4) controls. `w` is **not** "how much we trust the slow friend over the fast one." It's **how far out on the reach axis we let the fast friend go before the slow friend pulls it back toward ground truth.**

**What each owns (no overlap, no ranking):**

| | Fast friend (wild) | Slow friend (stable) |
|---|---|---|
| Owns | Reach, throughput, precise retrieval | Grounding, momentum, recovery, drift detection |
| Fails without the other | Drifts; no re-anchor point to return from | Never reaches; lossy stasis |
| Authoritative over | *Getting there* | *Whether we're still on course / where to reset* |

The slow friend is "authoritative" only in the narrow, correct sense: it is the **reference for drift and re-anchoring** precisely because it hasn't drifted. It is *not* authoritative about reach — that's the fast friend's job, and ceding it would collapse the polarity back into a single pole (and you'd get stasis). This is why the design keeps the slow friend a **soft scoping prior** and a **bounded weight**, never a hard mask or a boolean veto: a hard "slow wins" gate would be exactly the insensitivity this framing must avoid — it would demote the fast friend to a decoration.

**Why this matters for the build:** treat the two as **first-class peers with different jobs**, not as "primary model + fallback." A fallback framing smuggles in "slow is the backup / lesser" and biases the design toward treating slow-friend output as a degraded mode. It isn't — it's the *compass*. Build both to be load-bearing, and let `w` (and later scoped routing) express their balance rather than any fixed ranking.

**The shroom-sitter analogy.** Psychedelic users call the sober companion who grounds a trip a "trip sitter" or "shroom-sitter." That is exactly the slow friend's job under wild reach: not to *prevent* the trip, but to be the stable reference that says "yes, we started here, and this is what you actually saw vs. what the experience told you you saw." The fast friend is on a bad trip — brilliant, spontaneous, often right, occasionally catastrophically wrong about what's real. The slow friend is the one who can say "that building wasn't there before, let's check." Without the sitter, the trip is ungrounded velocity: you can't tell signal from hallucination after the fact. With the sitter, you get a receipt — *this* is what we saw, *this* is what was real, and here's the chain of evidence connecting them.

**Upper bound / lower bound weighting.** The two friends pull in opposite directions along an evidence axis:

- **Fast friend → upper bound.** It maximizes reach: "what could be true if we take this assumption?" It pushes toward the most ambitious interpretation, the widest plausible space, the fastest path to a novel answer. Its failure mode is *over-claiming*: asserting without sufficient grounding.
- **Slow friend → lower bound.** It guards the floor: "what do we actually have evidence for right now?" It pushes toward the minimum-viable-claim: what can you assert with confidence given what's been established? Its failure mode is *under-reaching*: refusing to go anywhere new because the evidence bar isn't met yet.

They are not in competition. They bracket a **credible range**: the fast friend defines how far out you *can* reach (upper bound), the slow friend defines how little you *must* claim (lower bound). The answer lives in between, and `w` controls where in that band you land. When divergence is low (the trip is coherent), `w` lets the fast friend operate near the upper bound — go wild, it's safe. When divergence rises (the trip is getting weird), `w` pulls the operating point back toward the lower bound — slow down, re-anchor, verify before proceeding.

**Mutual activation (neither is passive).** The relationship is not "fast acts, slow watches." Both *activate* the other:
- **Fast gets slow to act.** Without the fast friend's reach and throughput, the slow friend has nothing new to ground — it just re-confirms what it already knew. The fast friend generates the *material* (new claims, new context, new assumptions) that the slow friend then checks, scopes, and receipts. No fast → no trip → no need for a sitter.
- **Slow keeps fast from acting without sufficient evidence.** Without the slow friend's lower-bound guard, the fast friend acts on assumptions it can't verify — compounding errors, drifting into incoherence, losing the thread at high context. The slow friend says "not yet — here's what you'd need to establish first," and that *delays* the fast friend without *stopping* it. Bounded, not boolean: a nudge back toward evidence, not a hard veto.

**The receipt: the user's contribution must be verifiable.** This is the role that makes the whole thing *for the user*, not just for the model. The user participates in the loop — they provide intent, constraints, corrections, and domain knowledge. For that participation to mean something, they need a **receipt**: an audit trail showing (a) what they contributed, (b) how it shaped the fast friend's reach, (c) where the slow friend confirmed or corrected, and (d) the final grounded state. Without the receipt, the user can't distinguish "my input drove this result" from "the model hallucinated a plausible narrative that happens to include my words." The slow friend *is* the receipt: it's the stable, non-drifted record of what was established at each step, against which the fast friend's claims are checked. A user who can read the receipt — "you said X, the fast path explored Y, the slow friend confirmed Z because of your constraint W" — can take **credit** for their contribution with confidence, because the chain is auditable and the grounding is real.

This is why the slow friend must be *persistent* and *stable*: a receipt that drifts along with the trip proves nothing. The whole point is that it *doesn't* drift, so the user can look back and see what actually happened vs. what the experience claimed.

---

## 1. The reference architecture, mapped to slow/wild

The model's hidden layout is `12 × (3 × (GDN → MoE) → 1 × (QSA → MoE))` — three recurrent layers then one precise-attention layer, repeating (HF card). That is not incidental; it *is* the two-friend design.

| Tier | Component | Property | Slow/wild role |
|------|-----------|----------|----------------|
| Stable memory | **Gated DeltaNet** (`gdn_hybrid.tex`) | Fixed-size recurrent state `S_t ∈ R^{d_k×d_v}`, O(1) per token, bounded gates | **Slow friend** — the checksum / momentum |
| Precise retrieval | **Qwen Sparse Attention** (`qsa.tex`) | Bounded budget (512 blocks / 2048 tokens), MQA indexer selects scope first | **Wild friend** — precise but drift-prone at high context |
| Redundant store | **N-gram embedding** (`ngram.tex`) | 51B table, deterministically addressed, host-RAM offload + async prefetch | **CPU redundancy** — most "decisions" are free lookups |
| Cross-layer glue | **Gated Residual** (`residual.tex`) | Bounded positive read/write gates over `n_r=4` branches | Keeps the stream magnitude-bounded → FP8-storable |

### 1.1 GDN = the slow friend, made into a kernel

The gated delta recurrence (`gdn_hybrid.tex`):

```
S_t = α_t (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ ,   α_t ∈ (0,1), β_t ∈ (0,1)
```

"The decay `α_t` globally controls the lifetime of the existing state, whereas the delta term first estimates the value already associated with `k_t` and writes only the residual error." Two bounded gates: a **bounded forget** (`α_t < 1`) and a **bounded write** (`β_t < 1`). The state never grows with context. FlashQLA makes it 2–3× faster than FLA Triton forward — cheap enough to run on CPU. This is the *inertia* from `Computational Inertia Concept.md` made numerically real: "preserve computational intent without accumulating unbounded computational debt."

### 1.2 QSA = the wild friend, with a scoper already attached

Qwen Sparse Attention (HF card): 24 Q heads / 2 KV heads, head dim 256; an **MQA indexer** (4 query heads + 1 shared key head, head dim 128) selects a bounded budget of **512 blocks or 2048 tokens** before the precise attention runs *inside* that scope. The recipe reports up to 10.2× prefill / 6.6× decode attention speedups at 1M tokens. **This is the precedent for scoped routing (§3): a cheap scoper bounds where the expensive precise work happens.**

### 1.3 N-gram table = CPU-resident redundancy

`ngram.tex`: "deterministic addressing enables host-memory offloading and asynchronous prefetching." Placed at Layer 2 so "host-memory prefetching can overlap with the computation of the first layer." The vLLM recipe: `VLLM_PLE_CPU_OFFLOAD=1`, needs ≥51 GB host RAM. Because a known n-gram maps straight to table rows (multi-head hashing, no compute), **most local-context decisions are lookups, not decisions** — zero per-token FLOPs, living in system memory.

---

## 2. Bounded gates, not hard arbitrary gates

The paper's design philosophy, stated almost as a rule:

- `residual.tex` ablation, first finding: *"A sigmoid gate is better than tanh in both loss and training stability."* It notes this **recurs across GDN and attention** — sigmoid beats SiLU/tanh everywhere.
- GR read gate `σ(…) ∈ [0,1]` elementwise; write gate `s_i = 2σ(…) ∈ [0,2]`. Bounded, low-rank bottleneck, no matmul in the decision itself.
- The payoff line: *"The gates in GR, gated attention, and GDN all bound the magnitude of what is written into the stream, so residual values stay in a narrow range and are well matched to a low-precision format."* → they store the widened residual in **FP8**, halving bytes moved.

### Why bounded beats hard *in early multiplicative layers specifically*

1. **Early + elementwise = you can afford the full real-valued gate.** GDN/GR gates are O(d) or O(1) per token — no matmul in the decision. At layer 0, per token, you get a *continuous* decision for nearly free. A hard gate (binary mask / top-k cutoff / threshold reject) throws away that resolution and buys nothing in compute, because you still pay the surrounding multiply.
2. **Bounded ⇒ magnitude-limited ⇒ stable + low-precision.** A hard discontinuity amplifies gradient spikes and precision sensitivity; a bounded sigmoid gate can't blow up the stream. That is what keeps the recurrent state well-conditioned enough to sit in FP8 on CPU without drifting. Hard gates are where drift *starts*, not ends.
3. **"Decision isn't as expensive yet" = early softness is cheap, late hardness is costly.** An early-layer decision propagates through every downstream layer. Make it hard and arbitrary at the top and everything below inherits a committed binary choice — high consequence for low compute. A bounded gate keeps the option open ("partially pass / write / forget"), which is recoverable. By the time you're deep (or at the wild friend's precise retrieval), decisions *are* expensive and committed — so that's where the precise-but-drift-prone path lives, with the bounded early gates keeping the trajectory stable underneath it.

### The division already in the model

**Hard sparsity is reserved for compute** (the MoE router: 512 experts → 10 routed + 1 shared, a genuinely hard top-k), while **everything that accumulates state uses bounded soft gates** (GDN memory, GR residual, attention magnitude). You never put a hard arbitrary gate on the thing that must preserve momentum.

### The unifying rule

> **Bounded gates are the mechanism that keeps computational debt bounded.**

`α_t < 1` is a bounded forget; `β_t < 1` is a bounded write. Because every write to persistent state is magnitude-limited and decays, the system *cannot* accumulate unbounded error — "preserve intent without accumulating unbounded debt" made numerically stable instead of just a slogan. The inertia isn't a queue you defend; it's a recurrent state whose gates guarantee it stays finite and re-anchorable.

---

## 3. Scoped MoE routing ("the ask around method")

The one genuinely **hard** gate in the model is the MoE router (512 experts → top-10 + shared). The refinement: don't let that hard decision roam free over all 512 from the fast path's instantaneous, drift-prone state. **Scope it with stable memory first; do the precise top-k inside the scope.**

### The maintained bounded routing prior

The slow friend can't be in the per-token hot loop doing fresh work (that puts the slow thing on the critical path). So the scope must be a **maintained state**, updated incrementally by the bounded gates, read O(1) per token:

```
e_t ∈ R^E (or over expert groups)      # "which expert regions are in-scope
     = f(GDN_state_t, ngram_hits_t)        given stable context up to t"

per token:
  scope   = top-M of e_t                   # M > k, a bounded neighborhood
  experts = top-k(fast_router ⊗ prior(e_t), within scope)   # precise pick in-scope
```

- `e_t` is the **slow friend's routing prior** — a coarse expert-relevance map derived from the stable GDN state + deterministic n-gram hits.
- The fast path still makes the precise top-k, but *within* that bounded scope instead of over all 512.
- Because `e_t` is maintained incrementally (GDN update O(1), n-gram a lookup), it costs almost nothing to keep and read — the CPU-efficiency argument. The "decision about which experts to query" stays "within the scope of the slow mem processes" because the scope is *already computed* by stable memory, not recomputed from drifted state.

### Why this is bounded-not-hard (and why it kills drift)

- **Bounded, not arbitrary.** The scope is a soft prior / candidate set, not a hard mask. A hard mask reintroduces the cliff — if the prior is slightly off you zero out the right expert forever. A bounded prior with an expanded candidate set (M > k) keeps the option open: "these regions are likely relevant," then precise pick within them.
- **It fixes the exact drift failure.** At high context the fast path's instantaneous embedding softens (lost-in-the-middle, FP8 accumulation), so a free router wanders and activates the wrong experts — which compounds, because expert choice is high-consequence. But `e_t` comes from the GDN low-pass state that *hasn't* drifted. So the hard gate becomes **stable** because its candidate scope is anchored to stable memory. The wild friend gets a bounded leash on the one decision where going off-leash is most expensive.
- **The shared expert is the built-in floor.** Qwen already guarantees 1 expert is always active regardless of routing — a hard-guaranteed in-scope fallback. Generalize it: guarantee a bounded core set is always in scope (the stable experts); let the fast path choose the rest within the prior. That's the "no blackout" property applied to routing.

### N-gram makes most of it free

N-grams are **deterministically addressed** — a known bigram/trigram maps straight to rows, no computation. For common local patterns the expert scope isn't *decided* at all; it's **looked up** from stable memory. The slow friend pre-scopes routing for frequent cases with zero per-token decision cost; the fast path only makes a real top-k on genuinely novel tokens. "The decision stays within the scope of the slow mem processes" in its purest form: most decisions are just reads.

---

## 4. Drift-gated compaction / re-anchor

Replace a hard `if divergence > threshold: compact` (a cliff) with a **bounded confidence weight** `w ∈ [0,1]`:

```
loop:
  wild (GPU) runs fast, free-running, often-correct
  slow (CPU) maintains GDN state + n-gram checksum in parallel
  every N tokens: d = divergence(slow_summary, gpu_state)
  w = sigmoid-shaped(d)                     # bounded, not boolean
    w ≈ 1 : low divergence → wild runs free; slow just keeps its checksum
    w ↓   : rising pressure → route/compaction weight shifts smoothly toward re-anchor
    w → 0 : full re-anchor to the slow friend's ground truth, then take off again
```

The slow friend is the **reference clock / arbiter** precisely because it hasn't drifted — you compact *toward* it, not away from it. This inverts the usual "GPU is authoritative" assumption: at high context the cheap stable memory becomes the source of truth for re-anchoring. The "nope let's take a break and compact so we can take off again from a more similar understanding" is clock resync, made smooth by the bounded weight instead of a stop sign.

### Divergence metric options (cheapest first)

1. **State checksum** — project GDN `S_t` to a summary vector every N tokens; compare against the GPU's attention-derived representation of the same span (cosine / KL). Cheap, continuous, but couples to internal tensors.
2. **Logit divergence** — run both paths on the same prefix; measure next-token distribution distance at sampled checkpoints. Model-agnostic, but costs a forward pass on the slow friend.
3. **Semantic anchor checksum** — the slow friend answers a fixed set of anchor questions about established context ("what facts are locked in? what did we decide?"). When its answers stop matching the GPU's current behavior, the GPU has drifted from ground truth. A *semantic* (not just numerical) checksum; cheap because the slow friend is small.

### The editor node becomes bounded + early

The "meaningfully slows down the spontaneous output" role is a **bounded per-token correction applied early** (in the recurrent state), continuous — not a post-hoc hard reject of the wild friend's output. Soft brake, not a stop sign.

---

## 5. Three roles collapse into one slow-friend state

`e_t` / the GDN summary does all of it at once:

- **Fallback** — always warm, no blackout (the inertia doc's core). GPU drops → CPU carries momentum; routing stays sane on `e_t` alone.
- **Editor node** — second pass that bounds the spontaneous output by checking it against the stable summary, early and continuous. Fast writer + proofreader.
- **Load balancer** — route by *which device preserves the most useful momentum*, not just fastest: low-latency-tolerance / small context → CPU; high-throughput need → GPU. ("Accelerators provide throughput. CPUs provide continuity.")

All three are the same primitive: a cheap, always-on, stable reference that the fast path can be checked against and recovered from.

---

## 6. PESTI generalization: experts = compute units

Honest constraint first: **PESTI does not need to run Qwen3.8-Flash-Next itself.** The FP8 checkpoint is 172.78 GiB (TP4 on GB300 / TEP8 on H200; recipe). A dual-GPU box with ~32 GB VRAM cannot host it. But the *pattern* — cheap recurrent memory + bounded precise retrieval + host-RAM lookup table — transfers, and the **two-model split actually matches the slow/wild framing better** than the single fused model does:

- Small CPU model = checksum / scoping friend (stable, always-on).
- Big GPU model = wild friend (fast, drift-prone at high context).

This is a better fit for PESTI as a multi-backend substrate, and it's local-first.

The generalization that makes it *the load balancer* from the original idea:

- "Experts" = any sparse pool of **compute units** — GPU kernels, CPU fallback paths, sub-models, cached activations, even tools.
- The slow friend maintains `e_t` over that pool → decides which units are in scope per token/request.
- That is "overall load balancer," now with a concrete mechanism (maintained bounded routing prior) instead of a hand-wave about "route by momentum." And it's structurally CPU-friendly, because maintaining `e_t` is cheap stable work — what CPU does best.

### The PESTI design rule (testable invariant)

> **Route hard sparsity to compute (experts, attention budget). Route bounded soft gating to anything that persists (KV bookkeeping, GDN state, checksum confidence).**

That is the substrate-level form of "bounded gates keep debt bounded," and it makes the CPU slow friend the structurally correct home for all the bounded-gate logic.

---

## 7. What to prove before building (verification gates)

One load-bearing assumption: **does scoping the hard gate by stable memory actually reduce drift at high context?** It is testable with PESTI's existing conformance workflow, and PESTI already has seed evidence that the *drift signal* is real:

- `ROADMAP.md` Week 17 (measured): GPU-vs-oracle `max|d|` grows **5.6e-3 (layer 0) → 5.1e-2 (layer 23)**, prehead 1.4e-1, logits 5.0e-2; norm ratio 0.999–1.001, corr ≥ 0.999993, argmax identical. "The GPU residual is f16 tensor-core accumulation rounding … not a sync race or layout bug."
- Week 15 (measured): "23.9/20.7 max logit diff (f16 drift, **smooth per-layer growth, no structural jumps**)."

Both are exactly the *accumulation-drift* signature the slow friend is meant to detect and bound — smooth, depth-correlated, not a structural bug. So the probe is:

> Same prefix, growing `seq_len` (512 → 1024 → 2048+). Measure divergence between (a) a stable low-pass summary of the context and (b) the precise/GPU path's representation of that same span. Confirm the signal **grows with length** and is **smooth**. Then measure: does scoping routing/compaction by the stable summary keep activations closer to a low-context reference as length grows?

PESTI already has the tooling for this: per-layer capture (`capture_per_layer`, `dump_all_layers_gpu.rs`), `gpu_fallback_count()`, and the numpy-oracle diff path. The probe reuses it.

### Design cautions (same rule, restated)

- Keep the scope **soft** — prior + expanded candidate set (M > k), never a hard mask.
- Keep it **cheap to maintain** — project GDN state over expert *groups*, not all 512, if the pool is large.
- Keep compaction a **bounded weight**, not a boolean threshold.

---

## 8. Decision gates / nudges (see ROADMAP.md → Phase 5)

| Gate | Go criterion | No-go / fallback |
|------|--------------|------------------|
| **G1: Drift signal is real** | Divergence probe shows smooth, length-correlated growth on a real model | If flat → the drift premise is weak; revisit before building compaction |
| **G2: Bounded scope reduces drift** | Scoped router's expert activations stay closer to low-context reference as length grows vs free router | If no improvement → keep GDN state for fallback/editor only, drop scoped routing |
| **G3: Slow friend is cheap enough** | Maintaining `e_t` + checksum adds < budget (e.g. <5% step time) on CPU | If too slow → shrink summary dim / use n-gram-only scoping |
| **G4: Two-model vs fused** | Two-model split fits PESTI substrate + local-first better (expected) | Fused only if a single host can run the reference model |

---

*This is a direction, not a shipped change. The roadmap carries the open nudges; this doc carries the references.*
