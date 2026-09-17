/* RotSprite: three EPX rounds, then Pillow-exact nearest rotation.
 *
 * The reference (warlock.studio.inker.transform.rotsprite/epx) was measured at
 * 294 ms per mouse-move at 256^2 against a 16 ms gate -- superlinear, 22.5x the
 * cost for 4x the pixels -- and the drag runs on the frame thread on every
 * mouse-move (dev/measurements/2026-09-13-native-batch-10-candidates.md S5).
 * The arithmetic is integer compares and copies (EPX) plus one nearest sample
 * per output pixel, so a kernel is bit-exact by construction; the only
 * subtlety is that the sample has to reproduce Pillow 12.3.0's fixed-point
 * affine path (src/libImaging/Geometry.c, ImagingTransformAffine's nearest
 * branch) operand for operand, because that is what the numpy reference calls
 * through PIL.Image.rotate. A future Pillow whose affine_fixed rounds
 * differently is meant to fail tests/inker/test_rotsprite_native.py, not to
 * silently drift from this file.
 *
 * Scratch layout (see warlockc_rotsprite_u8 in warlockc.h for the contract):
 * the caller-owned `scratch` buffer is treated as two adjacent regions,
 *
 *   bufA = scratch[0 .. 16*h*w*channels)          -- exactly a 4x plane
 *   bufB = scratch[16*h*w*channels .. 80*h*w*channels) -- exactly an 8x plane
 *
 * and the three EPX rounds ping-pong through them: round 1 (src -> 2x) writes
 * into the head of bufB (which is not yet needed for anything else), round 2
 * (2x -> 4x) writes the whole of bufA, round 3 (4x -> 8x) reads only bufA and
 * writes the whole of bufB, discarding round 1's leftover bytes at bufB's
 * head. Every round reads one buffer and writes the other, so there is no
 * aliasing to reason about despite the sizes differing.
 */

#include "warlockc.h"

#include <math.h>
#include <string.h>

/* EPX's rule set for one pixel, given its four neighbours already resolved
 * (border replication happens in the caller, not here) -- shared by the
 * one-channel and four-channel rounds and by every column, border or not, so
 * there is exactly one place this arithmetic is written down.
 *
 * Only four equalities are needed per pixel, not the twelve a literal
 * transcription of the four "and not" rules would recompute (each rule shares
 * both of its terms with a neighbouring rule): LU = left==up, UR = up==right,
 * RD = right==down, DL = down==left, and the whole rule set is
 *
 *   q1 = LU && !DL && !UR   (up)      q2 = UR && !LU && !RD   (right)
 *   q3 = DL && !RD && !LU   (left)    q4 = RD && !UR && !DL   (down)
 *
 * which is transform.epx's docstring (A=up B=right C=left D=down) restated in
 * terms of the four pairwise comparisons instead of the eight in the literal
 * reading of it. */
static inline void epx_write_u8(uint8_t center, uint8_t up, uint8_t right,
                                uint8_t left, uint8_t down, uint8_t *out0,
                                uint8_t *out1, int64_t x) {
  int lu = left == up;
  int ur = up == right;
  int rd = right == down;
  int dl = down == left;
  int q1 = lu && !dl && !ur;
  int q2 = ur && !lu && !rd;
  int q3 = dl && !rd && !lu;
  int q4 = rd && !ur && !dl;

  /* A mask-and-xor branchless rewrite of these four selects was tried and
   * measured no faster: a 256^2 random mask stayed ~7.6 ms and a flat one
   * ~2.5 ms either way, so whatever MSVC does with this ternary here is
   * already not the noisy-mask cost -- see the 2026-09-17 measurement this
   * comment is next to keep anyone from trying the same rewrite again on the
   * strength of the theory alone. Left as the plain ternary. */
  out0[2 * x] = q1 ? up : center;
  out0[2 * x + 1] = q2 ? right : center;
  out1[2 * x] = q3 ? left : center;
  out1[2 * x + 1] = q4 ? down : center;
}

static inline void epx_write_u32(uint32_t center, uint32_t up, uint32_t right,
                                 uint32_t left, uint32_t down, uint8_t *out0,
                                 uint8_t *out1, int64_t x) {
  int lu = left == up;
  int ur = up == right;
  int rd = right == down;
  int dl = down == left;
  int q1 = lu && !dl && !ur;
  int q2 = ur && !lu && !rd;
  int q3 = dl && !rd && !lu;
  int q4 = rd && !ur && !dl;

  uint32_t v1 = q1 ? up : center;
  uint32_t v2 = q2 ? right : center;
  uint32_t v3 = q3 ? left : center;
  uint32_t v4 = q4 ? down : center;
  memcpy(out0 + (2 * x) * 4, &v1, 4);
  memcpy(out0 + (2 * x + 1) * 4, &v2, 4);
  memcpy(out1 + (2 * x) * 4, &v3, 4);
  memcpy(out1 + (2 * x + 1) * 4, &v4, 4);
}

/* One EPX / Scale2x round, one-channel plane (a selection mask) --
 * transform.epx transcribed pixel by pixel. `src` is h x w, tightly packed;
 * `dst` is 2h x 2w. Border pixels replicate themselves, matching
 * transform._neighbours exactly -- x = 0 and x = w - 1 are split out of the
 * loop below rather than handled with a per-iteration clamp, so the interior
 * loop (the overwhelming majority of columns at any size this module reaches)
 * has no branch in its index arithmetic for the auto-vectoriser to trip on. */
static void epx_round_u8(const uint8_t *src, int64_t h, int64_t w,
                         uint8_t *dst) {
  int64_t src_stride = w;
  int64_t dst_stride = 2 * w;

  for (int64_t y = 0; y < h; y++) {
    int64_t yu = (y > 0) ? y - 1 : y;
    int64_t yd = (y < h - 1) ? y + 1 : y;
    const uint8_t *row = src + y * src_stride;
    const uint8_t *rowu = src + yu * src_stride;
    const uint8_t *rowd = src + yd * src_stride;
    uint8_t *out0 = dst + (2 * y) * dst_stride;
    uint8_t *out1 = dst + (2 * y + 1) * dst_stride;

    epx_write_u8(row[0], rowu[0], row[(w > 1) ? 1 : 0], row[0], rowd[0], out0,
                out1, 0);
    for (int64_t x = 1; x < w - 1; x++) {
      epx_write_u8(row[x], rowu[x], row[x + 1], row[x - 1], rowd[x], out0,
                  out1, x);
    }
    if (w > 1) {
      int64_t x = w - 1;
      epx_write_u8(row[x], rowu[x], row[x], row[x - 1], rowd[x], out0, out1,
                  x);
    }
  }
}

/* Same round, four-channel plane (RGBA). Each pixel is loaded and stored as
 * one `uint32_t` via `memcpy` -- not a `(const uint32_t *)` cast, which would
 * be a strict-aliasing violation clang is entitled to miscompile -- so the
 * four-byte equality and copy is each one scalar op instead of a four-byte
 * loop, and `channels` never appears inside either loop nest: the dispatch
 * between this and epx_round_u8 happens once, in epx_round. Border columns
 * are split out the same way epx_round_u8 splits them. */
static void epx_round_u32(const uint8_t *src, int64_t h, int64_t w,
                          uint8_t *dst) {
  int64_t src_stride = w * 4;
  int64_t dst_stride = 2 * w * 4;

  for (int64_t y = 0; y < h; y++) {
    int64_t yu = (y > 0) ? y - 1 : y;
    int64_t yd = (y < h - 1) ? y + 1 : y;
    const uint8_t *row = src + y * src_stride;
    const uint8_t *rowu = src + yu * src_stride;
    const uint8_t *rowd = src + yd * src_stride;
    uint8_t *out0 = dst + (2 * y) * dst_stride;
    uint8_t *out1 = dst + (2 * y + 1) * dst_stride;

    {
      int64_t xr = (w > 1) ? 1 : 0;
      uint32_t center, up, right, left, down;
      memcpy(&center, row, 4);
      memcpy(&up, rowu, 4);
      memcpy(&right, row + xr * 4, 4);
      left = center;
      memcpy(&down, rowd, 4);
      epx_write_u32(center, up, right, left, down, out0, out1, 0);
    }
    for (int64_t x = 1; x < w - 1; x++) {
      uint32_t center, up, right, left, down;
      memcpy(&center, row + x * 4, 4);
      memcpy(&up, rowu + x * 4, 4);
      memcpy(&right, row + (x + 1) * 4, 4);
      memcpy(&left, row + (x - 1) * 4, 4);
      memcpy(&down, rowd + x * 4, 4);
      epx_write_u32(center, up, right, left, down, out0, out1, x);
    }
    if (w > 1) {
      int64_t x = w - 1;
      uint32_t center, up, right, left, down;
      memcpy(&center, row + x * 4, 4);
      memcpy(&up, rowu + x * 4, 4);
      right = center;
      memcpy(&left, row + (x - 1) * 4, 4);
      memcpy(&down, rowd + x * 4, 4);
      epx_write_u32(center, up, right, left, down, out0, out1, x);
    }
  }
}

/* Dispatch on `channels` once, outside both loop nests -- transform.epx
 * accepts a mask (h, w) or an RGBA plane (h, w, 4) alike, and this kernel
 * keeps that same two-shape contract without paying a per-pixel branch for
 * it. */
static void epx_round(const uint8_t *src, int64_t h, int64_t w, int channels,
                      uint8_t *dst) {
  if (channels == 4) {
    epx_round_u32(src, h, w, dst);
  } else {
    epx_round_u8(src, h, w, dst);
  }
}

/* Pillow's FLOOR/FIX macros (Geometry.c), transcribed rather than re-derived:
 * FIX turns a double coefficient into Pillow's 16.16 fixed-point form, and the
 * cast truncates toward zero for a non-negative value -- which is floor for
 * those -- while floor() itself handles a negative one. */
static int64_t fix16(double v) {
  double scaled = v * 65536.0 + 0.5;
  return (scaled < 0.0) ? (int64_t)floor(scaled) : (int64_t)scaled;
}

/* Explicit floor division by 65536. C's `>>` on a negative signed value is not
 * guaranteed to be an arithmetic (floor) shift by the standard this file is
 * built to (/std:c11), so this is spelled out rather than relied on -- the
 * incident this guards is a compiler that rounds toward zero instead, which
 * would shift every rotated pixel with a negative sample coordinate. */
static int64_t floordiv65536(int64_t v) {
  if (v >= 0) {
    return v >> 16;
  }
  return -(((-v) + 65535) >> 16);
}

/* Where output sample (X, Y) lands in the 8x plane, or "nowhere" -- the one
 * piece of the sampling loop below that channels never touches, factored out
 * so the two channel-specific loops share it instead of each restating the
 * fixed-point arithmetic. Returns 0 (leave the destination at its memset
 * zero) or 1 with `*xin`/`*yin` filled. */
static int rotsprite_sample(int64_t X, int64_t Y, int64_t A0, int64_t A1,
                            int64_t A2, int64_t A3, int64_t A4, int64_t A5,
                            int64_t w8, int64_t h8, int64_t *xin,
                            int64_t *yin) {
  /* Exact restatement of Pillow's incremental per-row stepping (xx = A2,
   * xx += A0 each column, A2 += A1 each row) as one direct sum: both are
   * integer sums of the same terms, and X, A0 etc are bounded far below
   * where an int64 product could overflow at any size this module reaches
   * (ROTSPRITE_MAX_PIXELS caps w, h at 512). */
  int64_t xx = A2 + X * A0 + Y * A1;
  int64_t yy = A5 + X * A3 + Y * A4;
  int64_t x = floordiv65536(xx);
  if (x < 0 || x >= w8) {
    return 0;
  }
  int64_t y = floordiv65536(yy);
  if (y < 0 || y >= h8) {
    return 0;
  }
  *xin = x;
  *yin = y;
  return 1;
}

int32_t warlockc_rotsprite_u8(const uint8_t *src, int32_t h, int32_t w,
                              int32_t channels, double a0, double a1,
                              double a2, double a3, double a4, double a5,
                              int32_t out_h, int32_t out_w, uint8_t *scratch,
                              size_t scratch_len, uint8_t *out) {
  int64_t hw = (int64_t)h * (int64_t)w * (int64_t)channels;
  int64_t size_4x = 16 * hw;
  int64_t size_8x = 64 * hw;
  size_t needed = (size_t)(size_4x + size_8x);

  if (scratch_len < needed) {
    return -1; /* the caller falls back to numpy rather than guessing bigger */
  }

  uint8_t *bufA = scratch;           /* holds the 4x plane, exactly */
  uint8_t *bufB = scratch + size_4x; /* holds the 8x plane, exactly */

  epx_round(src, h, w, channels, bufB);                              /* -> 2x, head of bufB */
  epx_round(bufB, 2 * (int64_t)h, 2 * (int64_t)w, channels, bufA);   /* -> 4x, all of bufA */
  epx_round(bufA, 4 * (int64_t)h, 4 * (int64_t)w, channels, bufB);   /* -> 8x, all of bufB */

  const uint8_t *big = bufB;
  int64_t h8 = 8 * (int64_t)h;
  int64_t w8 = 8 * (int64_t)w;
  int64_t big_stride = w8 * (int64_t)channels;

  /* The affine coefficients, Pillow's own fixed-point recipe
   * (Image.rotate -> Image.transform -> ImagingTransformAffine):
   * A2/A5 fold in the half-pixel-centre offset Pillow's own per-row loop
   * would add via (x0 + 0.5). */
  int64_t A0 = fix16(a0);
  int64_t A1 = fix16(a1);
  int64_t A3 = fix16(a3);
  int64_t A4 = fix16(a4);
  int64_t A2 = fix16(a2 + a0 * 0.5 + a1 * 0.5);
  int64_t A5 = fix16(a5 + a3 * 0.5 + a4 * 0.5);

  memset(out, 0, (size_t)out_h * (size_t)out_w * (size_t)channels);

  /* `channels` dispatched once, outside the loop nest -- the same shape as
   * epx_round's dispatch above, and for the same reason: a per-pixel branch
   * here is a per-pixel branch on every one of out_h * out_w samples. */
  if (channels == 4) {
    for (int64_t j = 0; j < out_h; j++) {
      int64_t Y = 4 + 8 * j;
      uint8_t *out_row = out + j * (int64_t)out_w * 4;
      for (int64_t i = 0; i < out_w; i++) {
        int64_t xin;
        int64_t yin;
        if (!rotsprite_sample(4 + 8 * i, Y, A0, A1, A2, A3, A4, A5, w8, h8,
                              &xin, &yin)) {
          continue; /* stays the zero fill memset above */
        }
        uint32_t v;
        memcpy(&v, big + yin * big_stride + xin * 4, 4);
        memcpy(out_row + i * 4, &v, 4);
      }
    }
  } else {
    for (int64_t j = 0; j < out_h; j++) {
      int64_t Y = 4 + 8 * j;
      uint8_t *out_row = out + j * (int64_t)out_w;
      for (int64_t i = 0; i < out_w; i++) {
        int64_t xin;
        int64_t yin;
        if (!rotsprite_sample(4 + 8 * i, Y, A0, A1, A2, A3, A4, A5, w8, h8,
                              &xin, &yin)) {
          continue;
        }
        out_row[i] = big[yin * big_stride + xin];
      }
    }
  }
  return 0;
}
