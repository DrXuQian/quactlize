# Q4 indexed MoE S1: PPU result review

The corrected v2 port passes **108/108 cases and 1,728 recipe correctness
checks** on PPU. The old signed-versus-unsigned helper error is closed for
this gate. This admits the operator to the combined SIMT/TC comparison; it
does not change production selection or establish parity with another route.

Archive: `q4-moe-s1-ppu.CQTor0.results.tgz`, SHA256
`d5e519561387fe03bcd2c15002428b882133de071d677f6d63936bf8039801a1`.
Reviewed source `e700000`, package `prebuilt/ppu0010/q4-moe-s1-v2`.
The [review receipt](measurements/q4_moe_s1_20260913/review.json) contains
authority, numerical checks, timing caveats and imported ACU counters. The
[108-row table](measurements/q4_moe_s1_20260913/summary.tsv) retains each
winner, runner-up and round-spread warning. No old failed timing was reused.

## Scope and timing

Q4 only, E=256, topk=8, tokens1..8. Each token selects eight expert rows;
token1/top8 is not dense M8 on one weight. Both shared gate/up activation
(`channels=1`) and slot-specific down activation (`channels=8`) pass.
Dispersed cases use 8..64 distinct experts, one row per expert. Additional
token8 clustered cases reuse eight experts, eight rows per expert.

All times below are microseconds per **complete indexed S1 call**, not per
token. GPU IDs, activation lookup, expert weight lookup and direct FP32
output are inside one kernel. No standalone gather, scatter, directory
builder or inter-CTA reducer is required. Top-k computation, activation
functions and full model/chain execution are outside this scope.

| N | K | Token1 shared A | Token1 slot A | Token8 dispersed, slot A | Token8 clustered, slot A |
|---:|---:|---:|---:|---:|---:|
| 512 | 2048 | 7.995 | 8.028 | 33.473* | 27.047 |
| 512 | 3072 | 11.500 | 11.524 | 48.979* | 38.896 |
| 2048 | 512 | 7.753 | 7.805 | 34.762 | 29.099 |
| 3072 | 512 | 9.656 | 9.639 | 50.629 | 42.866 |
| 1024 | 2048 | 11.409 | 11.580 | 65.155* | 52.416 |
| 1024 | 3072 | 16.665 | 17.115 | 93.468* | 74.487 |

The N1024 cases represent merged gate/up output widths, not certification of
the entire fused chain. Q4_K_M models can have Q5/Q6 down weights; this table
does not admit their SIMT readers.

`*` Selected-recipe round span exceeds 5%. Across the full gate, **18/108**
cases have this warning, predominantly larger dispersed-token cases. The
maximum span is 9.34%; the median span over all cases is 0.228%. These are
valid numerical cases with provisional timing, not proof of GPU interference
or a reason to rerun the other 90 cases. There is no exclusive-device-use
receipt. Reconfirm the noisy subset in the next matched route sweep.

There are 16 screening recipes per case, five samples each. The top two get
six alternating rounds of fifteen samples; all 28,080 samples are finite,
and all medians/order/selected recipes were independently recomputed.
75/108 finalist pairs differ by at most 5%, including 23 pairs within 1%.
Do not turn every small recipe difference into a production rule. These are
bounded measured winners, not proof of global optimality.

The run took 251.61 seconds including fixture, correctness and profiling.
Timing excludes setup, first launch and graph upload. Complete rotating-ring
traversals touch at least 144 MiB of **active-expert weights**, versus the
64 MiB L2 returned by device attribute38 and independently by ACU. The ring
does not count the unused experts as cache pressure.

## Correctness closure

The maximum condition-normalized original-GGUF dot error is
**5.1948897e-4**, below 0.005. It comes from META's per-weight FP16
reconstruction. MEDIUM/REUSE's maximum FP32 group-affine error is
**2.2284600e-7**. Both accumulate/output FP32; F32 activations are rounded
to F16 in registers. These two dequant arithmetic contracts are not bitwise
identical, and neither is an activation-quantized Q8 path.

All recipes pass F32-rounded/F16 equality, indexed/compact equality,
indexed/rebased-dense equality, zero-A, zero-code rejection, invalid IDs,
output/padding guards and changed-GPU-ID graph replay. The overlapping
N512/K2048 META cases also match the old immutable dense DSO exactly.
The 0.609534 failure was reproduced independently and fixed before this run;
see the [causal regression](../Q4_MOE_S1_NUMERIC_FIX.md).

All 134 hashed result files, 20 package source hashes, 14 harness hashes and
six payload identities match. The harness checks the four runtime library
digests before running; the archive does not contain an independent loader
path dump. The erroneous `properties_sm=1` field is not used as a hardware
fact: the imported reports identify **72 CUs** and 64 MiB L2.

## ACU: what improved and what still needs a comparison

All six reports import successfully, with 12 exact indexed-kernel launches
and matching recipe, F32-input specialization, grid, block and shared-memory
geometry. All report zero shared-memory bank conflicts. ACU is forced-cold
replay; its durations are **not** the rotating event times above.

For N512/K2048, token1 versus clustered token8 still touches eight experts:

| Metric | Token1, W8/P8 | Token8 clustered, W2/P8 |
|---|---:|---:|
| DRAM read, decimal MB | 4.759 | 5.426 |
| L1/L2 traffic, decimal MB | 10.132 | 96.748 |
| Vector load instructions | 18,432 | 141,312 |
| Registers/thread | 82 | 100 |
| Shared bytes/CTA | 1,024 | 256 |
| Achieved occupancy | 21.89% | 42.95% |
| Rotating complete-call event time, us | 7.995 | 27.047 |

The extra rows do **not** fetch eight full copies of B from DRAM. They still
repeat substantial on-chip reads and decode/accumulation work because this
implementation assigns one row per CTA group and has no explicit cross-row
B reuse. The two columns select different warp counts, so they are not a
single-variable test. This motivates retaining TC for the multirow sweep,
not concluding that SIMT must win every token count or that memory-pipe-busy
alone explains latency. Shared memory is only 128..1,024 bytes in these
profiled winners; it is not the earlier TC shared-memory-capacity issue.

The 1,728 recorded address models were rederived. At N512/K2048, REUSE
W8/P8 has four adjacent 16-byte B requests per lane cohort: full unique-byte
footprint utilization at 32/64-byte granularity, 50% at128. Cooperative
metadata requests fill all three granularities. Its noncooperative F32 A
instruction still has fourfold duplicate lane requests. The N512/K3072 META
winner has worse B footprints, but remains faster in this bounded cohort;
coalescing alone is not a timing verdict. The source-bound native manifests
retain LOP3/half2 fast dequant plus FP32 accumulation; static counts are not
dynamic instruction counts.

For token1 the weight-bytes/event-time bandwidth model is 590..849 GB/s,
or 21.9..31.5% of the user's 2700 GB/s roof. This is not ACU-measured DRAM
utilization, nor a same-cohort comparison with the raw-GGUF reference/TC.

## Next admission boundary

1. Add these exact measured recipes to the prior-shape SIMT/TC sweep without
   dropping previous TC winners. Compare necessary adapters and real
   Split-K reduction inside the complete-call interval.
2. Reconfirm noisy/near-tie cases there, then derive a compact heuristic.
   No current result justifies copying the dense M thresholds into MoE.
3. Preserve existing gate/up fusion and mixed SIMT/TC chain behavior when
   enabling the direct reader in llama.cpp. The present chain entry rejects
   direct GEMV; simply selecting this kernel could otherwise disable fusion.

No production selector, kernel or offline format was changed by this review.
