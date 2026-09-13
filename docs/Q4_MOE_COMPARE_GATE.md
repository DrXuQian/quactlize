# Cold Q4 indexed SIMT versus grouped TC

This is the next **MoE operator comparison**, not another S1-only rerun, a
full model benchmark, or the later all-shape dense/grouped policy campaign.
The [preceding S1 gate](Q4_MOE_S1_RESULTS_20260913.md) passed all108 cases.
The existing [dense M2..8 comparison](Q4_SMALLM_RESULTS_20260913.md) is retained;
its timings are not mixed with this run to manufacture a new route verdict.

## Compared work

- Q4, six N/K families:512/2048,512/3072,2048/512,3072/512,1024/2048,1024/3072.
- E256, top8, tokens1..8, shared gate/up A and slot-specific down A;108 cases,
  including clustered token8. Both arms use identical original-GGUF weights,
  IDs, padded F32 inputs and padded F32 output order.
- SIMT retains the **two actual measured finalists for each case** from the
  preceding gate, including its18 noisy cases. No unmeasured global recipe
  replaces a previous winner. This is a two-finalist replay, not all16 recipes.
- TC includes every historical decode anchor in these families at total
  rows<=64/max expert rows<=8, the current policy's selected parent and the
  six-parent TM8/TM16 decode neighborhood. Ten unique compiled parents.
- S1/2/4/8 are runtime axes. Persistent parents also test capacity/balanced
  grids at B1/2/4 and retain the exact current-policy grid recipe. K-tile and
  pipeline exclusions are printed as structural rows, never fake timings.
- TC final output rounds through F16; SIMT group-affine output remains F32.
  Both round caller F32 activations to F16 and accumulate FP32. Independent
  GGUF error<0.005 is required; unequal legal rounding is not raw-bit equality.

The TC complete-call timing includes GPU routing/gather, descriptors and
directory, GEMM and ordered reduction/output scatter where required. It
never receives a host-prepared expert histogram during measured execution.
There is no timed H2D/D2H, JIT, graph upload or weight packing.

The existing fused indexed TC ABI accepts at most32 routed rows. Tokens1..4
therefore execute that exact native path. Tokens5..8 use a **development
GPU endpoint adapter** (rank/gather, unchanged device-only TC call including
metadata/directory/reducer, then scatter). The adapter supports64 rows and
uses the production rank functions; it moves A, not B. Its costs are inside
events. This does not widen the shipping fused ABI or claim the experimental
fallback is the optimal production adapter. The two scopes are explicit in
each result. Full MoE chain/router/activation fusion remains outside this gate.

## Execution and recovery

All six shapes use a rotating ring sized from only eight active experts,
at least2.25x verified L2, and whole-ring graph traversals. Each case starts
in a fresh process. Five samples screen each candidate; the top two of each
arm receive six alternating rounds of15 samples. The first graph launch is
excluded. Full wall-time ETA is updated from completed cases, advisory only.

Runtime/numeric failures are distinct from structural exclusions. Successful
cells are checkpointed after measurement. A failure restarts that case in a
fresh process, skips its recorded failed cell and finishes remaining cells;
the case remains INCOMPLETE. Explicit resume retries failed cells while
retaining successful ones. Changed finalists invalidate only the applicable
confirmation set, retaining its previous evidence separately. Another case's
success is never discarded. Output/workspace guards and changed-GPU-ID replay
are checked before timing, and rotating-graph outputs are rechecked afterward.

Default ACU captures both selected arms for token1/shared A and
token8/clustered slot A in every shape:12 reports,24 profiled complete calls
(TC calls contain multiple kernels). These forced-cold replay durations are
not substituted for the rotating event measurements. Report failure does not
invalidate the completed numerical/event case. `ACU=0` skips profiling.

## Box command

```bash
cd /sim/eec/shared/junfu.qx/quactlize &&
git pull --ff-only origin develop &&
RESUME_RUN= \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
CUDA_VISIBLE_DEVICES=0 \
bash tools/run_q4_moe_compare_ppu_box.sh
```

Only required payloads are fetched through Git LFS. **No compilation/JIT on
box.** The new TC/helper payload is about5.2MB; the existing S1 payloads are
reused. Failures preserve the caller's Docker shell. Send the printed
`q4-moe-compare-ppu.*.results.tgz` archive back, including ACU reports.

For a restart, set `RESUME_RUN` to this comparison's existing run directory.
Do not point it at the preceding S1-only directory: the schemas differ.
For a first runtime calibration, the underlying runner also accepts
`--case N K tokens channels clustered`; the full command prints ETA after
every case without imposing a time limit. Exact campaign duration has not
yet been established by a PPU run.

Local verification covers parent/winner recall, exact sample/confirmation
denominators, checkpoint/retry negatives,64-row rank/offset contracts,
complete-call accounting and PPU compilation. Device correctness/timing of
this new comparison, especially the64-row adapter, remain the box gate's
job; a host PASS is not device admission. Production selection is unchanged.
