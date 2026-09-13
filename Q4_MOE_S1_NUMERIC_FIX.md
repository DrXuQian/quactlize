# Q4 MoE S1: signed helper transplanted into an unsigned reader

Affected source: `be2bc98`, prebuilt `q4-moe-s1-v1`. This is an integration
error in the new S1 reader port, not a change to canonical weight bytes or a
PPU-versus-NVIDIA hardware limitation. Do not use the failed affine timings.
Earlier dense tuning images are unchanged and are not affected by this port.

## Causal difference

| Reader | Codes required | Scale/zero interpretation |
|---|---|---|
| META | `q - 8` | half reconstruction with `zero + 8*scale` |
| MEDIUM / REUSE | `q` | FP32 group affine `scale*dot(q,A) - min*sum(A)` |

The port copied the outer kernel bodies correctly, but obtained `codes()`
from the original `q4_native.cuh`, before the dense generators applied their
unsigned-code and fast-dequant transformations. All three readers therefore
received the signed helper. The affine readers subtract an uncompensated
`8 * sum_g(scale_g * sum(A_g))` from each output.

For the actual token1/channels1 fixture at N1024/K3072 (experts selected as
`3 + 17*slot`), independently computing that missing term from original GGUF
metadata gives normalized error **0.609533505841744**. The box reported
**0.609534**. This reproduces the magnitude without changing a schedule,
thread mapping, route or format and distinguishes the fault from an index or
CUDA-event problem.

## Fix and invariants

`codes<Slot,Bias>` now requires an explicit compile-time Bias: META uses 8;
MEDIUM and REUSE use 0. The helper is derived from the **fully generated,
admitted dense source**, including its LOP3 + half2-FMA fast dequant, not from
the earlier generic header. There is no new runtime branch or correction
pass. A/B/metadata addresses, loads, CTA count, shared partials, barriers,
FP32 dot order and output addressing are unchanged.

The new PPU images live in `prebuilt/ppu0010/q4-moe-s1-v2`; v1 is retained as
the exact historical negative, not overwritten or admitted. Run the box gate
in a fresh result directory, not by resuming a v1 receipt.

## Regression boundary

The original host tests checked the new generator against its own copied
header. That was insufficient: both shared the same helper mistake. Added
checks compare each family with the final dense helper, independently verify
all 65,536 b16 words across four slots and both biases, and require the
reported 0.609534 error to be reproduced by the legacy missing term.

`dev/gemv_cuda/prepare_moe_s1_check.py` exports source-bound old/fixed
diagnostic modules for RTX5070. The only platform adapter is a development
copy substituting CUDA runtime names; the real reader bodies, indexed API,
activation view and validation are retained. No NVIDIA compatibility code is
added to the PPU implementation.

The CUDA check covers six shapes, 16 recipes, token1..8, channels1/8, both
F16/F32 inputs, 16 distinct resident experts and padding guards. Old affine
readers must be RED, old META and all repaired readers GREEN, with exact
F16-versus-rounded-F32 output. PPU admission remains separate: its gate
covers E256, compact/indexed/dense equality, changed GPU IDs on graph replay,
negative inputs, cold timing and ACU. A CUDA PASS does not certify PPU timing.

## RTX5070 device result, 2026-09-13

[Source/image-bound review](docs/measurements/q4_moe_s1_cuda_20260913.json):
6,144 unique cells checked against the exact Cartesian denominator.
All 3,072 fixed cells pass; all 2,496 legacy affine cells fail as expected;
576 legacy META controls pass. All 3,072 F16/F32 pairs are bit-exact,
including the old negative, and every output/padding guard remains intact.

| N | K | Fixed max normalized error | Fixed affine max error |
|---:|---:|---:|---:|
| 512 | 2048 | 2.079953e-4 | 2.242089e-7 |
| 512 | 3072 | 1.786241e-4 | 2.127929e-7 |
| 2048 | 512 | 4.801415e-4 | 1.571905e-7 |
| 3072 | 512 | 5.243794e-4 | 1.638114e-7 |
| 1024 | 2048 | 2.285763e-4 | 2.291448e-7 |
| 1024 | 3072 | 1.828046e-4 | 2.147590e-7 |

Local host regressions: **164 passed**. All six corrected PPU modules compile
in 28.9 seconds; 192 native input/recipe specializations pass symbol and
fast-dequant/FP32 ISA checks. The PPU gate must still be rerun; it does not
inherit NVIDIA correctness or performance admission.
