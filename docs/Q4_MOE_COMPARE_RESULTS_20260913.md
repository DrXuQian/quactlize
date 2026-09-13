# Q4 indexed MoE: matched SIMT/TC results

**108/108 cases pass. SIMT wins 98; TC wins 10.** All 96 dispersed-routing
cases favor SIMT. The ten TC wins are token8 clustered cases, where eight
tokens reuse the same eight experts; each expert has eight rows. The small
N512/K2048 clustered cases still favor SIMT. None of the arm differences
is within 5% in this cohort. This is a useful routing boundary, not a rule
that all token counts up to eight should select SIMT.

Archive: `q4-moe-compare-ppu.2EHMoT.results.tgz`, SHA256
`24333529baeb7bdaf6c9bfc8a0ee874926abc8a12856b056ce23b16343557943`.
Source `d9a3612`, TC package `prebuilt/ppu0010/q4-moe-compare-v1` and unchanged
SIMT package `prebuilt/ppu0010/q4-moe-s1-v2`.
[Review receipt](measurements/q4_moe_compare_20260913/review.json),
[108-row table](measurements/q4_moe_compare_20260913/summary.tsv), and
[all imported ACU metrics](measurements/q4_moe_compare_20260913/acu-metrics.json.gz).
The reproducible reviewer is `dev/gemv_ppu/review_moe_compare.py`.

## Complete-call latency

Times are microseconds per indexed operator, **not per token**. E256/top8,
Q4 only, cold rotating active weights, F32 indexed endpoints. Both arms
round A to F16 and accumulate FP32. TC includes its GPU preparation,
directory/metadata, GEMM, actual reduction and output scatter as applicable;
the SIMT S1 reader does all indexed lookup and direct output in one kernel.
Top-k computation, gate activation, the complete MoE chain and model are
outside this comparison.

Token1 below uses shared gate/up A. Clustered token8 below uses slot-specific
down A. The two input conventions were both tested for every case.

| N | K | Token1 SIMT | Token1 TC | Token8 clustered SIMT | Token8 clustered TC |
|---:|---:|---:|---:|---:|---:|
| 512 | 2048 | 8.012 | 20.268 | 27.033 | 29.862 |
| 512 | 3072 | 11.415 | 22.704 | 38.805 | 33.172 |
| 2048 | 512 | 7.773 | 20.328 | 29.120 | 25.748 |
| 3072 | 512 | 9.619 | 22.472 | 42.803 | 28.010 |
| 1024 | 2048 | 11.333 | 24.376 | 52.312 | 34.126 |
| 1024 | 3072 | 16.613 | 27.592 | 74.431 | 39.636 |

The N1024 cases cover merged gate/up widths; this does not certify full
gate/up chain fusion. A Q4_K_M model can still have Q5/Q6 down weights.
No MoE raw-GGUF reference was measured in this gate; the preceding dense
raw-reference results cannot be relabeled as MoE/model parity.

Tokens1..4 use the existing native fused indexed TC path. Tokens5..8 use
the disclosed development 64-row GPU adapter plus unchanged device-only
TC entry. All adapter costs are included; this is not a claim that the
development fallback is optimal or that the shipping fused ABI now supports
64 rows. It also means the latency change from token4 to token5 crosses an
adapter boundary: do not fit a smooth token-only curve through it.

## Validated evidence

- 216 SIMT and 3,060 TC successful correctness/screen records. Another 1,224
  TC cells are explicit structural exclusions, not numeric failures or
  timings. All successful independent-GGUF checks, zero-code negatives,
  zero-A controls, guards and changed-GPU-ID replay controls pass.
- Maximum condition-normalized error: SIMT `1.3818633e-4`, TC `8.0763017e-4`,
  both below `0.005`. TC preserves its F16 output boundary before F32
  scatter; SIMT retains its reader-specific dequant rounding and F32 output.
  This is not a cross-arm raw-bit identity claim.
- 2,592 confirmation records and 55,260 finite samples. The actual prior
  SIMT winner/runner and historical/current TC parents were retained. Each
  arm's best is recomputed from six rounds of fifteen samples after the
  five-sample screen. These are bounded measured winners, not a global
  optimality proof.
- All 363 result-file hashes, 663 TC build-source hashes, package payload
  identities and current harness identities match. The harness checks the
  runtime library hashes before execution; no independent loaded-path dump
  is present in the archive.
- Device attribute38 and all imported profiles agree on 64 MiB L2. ACU
  reports 72 CUs. The erroneous property-struct `properties_sm=1` is not used
  as hardware truth. The ring counts only eight consumed expert slices and
  traverses whole rings exceeding 2.25x L2. Setup, first use, JIT and graph
  upload are outside resident event timing.

Two **TC** winners have round-span warnings: N512/K2048 token4, shared A
8.90% and slot A6.61%. SIMT is about62–63% faster there, so these are not
near-tie route reversals. All other selected-arm spans are <=5%. The
preceding S1 gate's 18 noisy cases were remeasured; none is a noisy selected
SIMT winner in this cohort. All648 paired confirmation-round comparisons
retain their case's winning arm; the smallest absolute paired difference
is7.43%. Preserve distributions when fitting recipes.

Total runner wall time, including setup/correctness/ACU, was1,880.996s
(31.35min). ACU warns about one other Python context owner. Its identity is
retained in the review; the parent runner probes the device before spawning
children, which can explain a resident context. This is not proof of either
exclusive use or concurrent inference, and no idle-device receipt was
collected.

## ACU interpretation

All12 reports import, with64 kernel launches. The selected SIMT template,
F32-input specialization and grid/block/shared geometry match. TC grid,
block/shared geometry and complete-call kernel sequences also match the
selected recipe, including real reducers. ACU is forced-cold kernel replay;
its individual durations must not replace rotating event times or be summed
as a measured graph end-to-end latency.

For N512/K2048 token1, ACU records:

| Component | Profiled kernel time, us | DRAM reads, decimal MB |
|---|---:|---:|
| SIMT indexed S1 | 9.458 | 4.759 |
| TC fused prepare | 9.075 | 0.015 |
| TC producer | 13.326 | 4.942 |
| TC reduce/scatter | 1.997 | 0.070 |

Across all six token1 profiles the fused prepare is8.94–9.41us. It moves A
and derives routing/metadata, **not B weights**. Source inspection shows
the indexed entry launches `(ceil(K/256), routed_rows)` CTAs, each recomputing
the route/ranks while only one writes the complete directory. The core TC
kernel is also slower than the SIMT kernel in these token1 profiles; the
whole gap is not merely extra kernel launches. Further TC preparation work
remains a separate optimization, not a reason to subtract its cost here.

At N1024/K3072, clustered token8, the reverse compute trend is clear:

| Metric | SIMT S1 | TC producer |
|---|---:|---:|
| Profiled kernel time, us | 75.718 | 23.415 |
| DRAM reads, decimal MB | 15.245 | 14.965 |
| L1/L2 traffic, decimal MB | 279.232 | 31.439 |
| Vector load instructions | 425,984 | 50,176 |
| Achieved active occupancy, % | 58.56 | 22.06 |

This supports repeated on-chip reads/decode in the per-row SIMT reader,
despite similar DRAM bytes. TC reuses the weight work across expert-local
rows. Higher occupancy alone did not make SIMT faster. The development
rank/gather is itself14.732us in this profile, and the full TC call still
wins after its measured preparation/reduction/scatter costs. The raw ACU
archive includes bank-conflict and memory-stall counters; they are not
wall-time percentages or a unique causal attribution.

## Next work

1. Keep SIMT eligible for token1 and the measured sparse expert-local-M1
   cases, retain TC for concentrated multirow cases. Do not infer the
   unmeasured partial-collision boundaries (expert-local rows2..7) from
   the two extreme routers, or introduce D2H histogram reads to select a
   timing winner.
2. Complete the earlier requested prior-shape dense/grouped GEMV+TC sweep,
   retaining earlier TC winners (including the dense TM8/S4 gap), then
   update the compact heuristic. This six-family operator gate does not
   finish that larger sweep.
3. Preserve gate/up fusion and mixed SIMT/TC chains when enabling the reader
   in the production JIT/dispatch and llama.cpp. Recheck route traces,
   independent numerics and steady-state model latency against native.
   The current chain entry's direct-GEMV exclusion remains a blocker to
   simply switching the selected operator.

No production selector, kernel, ABI, offline format or llama.cpp wiring was
changed by this review.
