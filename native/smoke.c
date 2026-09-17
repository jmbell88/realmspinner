/* Flourish's smoke primitive: one blob's per-pixel work (distance plane,
 * optional fbm raggedness blend, coverage, over_into) --
 * warlock.studio.inker.flourish.prims.smoke.render's per-blob loop body plus
 * the noise.value2d/fbm and prims.fbm_plane/over_into helpers it calls.
 *
 * Measured at 388 ms/frame for an 80-blob smoke layer against the Flourish
 * bake path's 100 ms gate (batch 10 S4,
 * dev/measurements/2026-09-13-native-batch-10-candidates.md), almost all of
 * it per-blob numpy dispatch over a window of a few thousand pixels; batch
 * 11's prototype got that to 80.6 ms, bit-identical, by fusing the whole
 * per-pixel chain into one call per blob. All the scalar prep that does not
 * vary per pixel (blob centre, radius, the fbm offsets, the alpha*coverage
 * multiplier, the rgb triple) stays in Python, computed with the exact same
 * numpy expressions the shipped code already uses -- this file only has to
 * get the per-pixel array math bit-identical: distance -> fbm blend ->
 * coverage -> over_into.
 *
 * Precision notes (verified against a running numpy 2.x via NEP 50, not
 * assumed -- see the batch 11 prototype this was transcribed from):
 *   - px, py are numpy.float64 (np.cos/np.sin of a python float returns a
 *     real np.float64 scalar, which is NOT "weak" under NEP 50), so
 *     (win.x - px) upcasts the float32 window coordinate to float64.
 *     Therefore d, d2, cov, cov2 are all float64 here, even though the
 *     arrays around them are float32 in the shipped code. Only the fbm
 *     noise plane and rgb stay float32 throughout.
 *   - over_into does `out_rgb *= (1-cov)[...,None]` and
 *     `out_rgb += rgb*cov[...,None]` as two SEPARATE in-place ops on a
 *     float32 array; each in-place op is computed at the common (float64)
 *     precision and rounds to float32 once it stores. So out_rgb needs
 *     *two* float32 roundings, not one fused double compute. out_a's `+=`
 *     is a single statement with the whole RHS built in double first, so it
 *     needs only one rounding.
 *
 * No allocation: every scratch plane this needs (the fbm coarse grid, its
 * upsampled-and-blurred raster, and the box-blur's row/column work buffers)
 * is caller-owned, sized by native.py's smoke_blob() from the window and the
 * scale before the call. -1 means the scratch was too small and nothing has
 * been written to out_rgb/out_a -- the caller falls back to numpy rather
 * than guessing a bigger buffer, same contract as warlockc_contours,
 * warlockc_bvh_build and warlockc_rotsprite_u8.
 */

#include "warlockc.h"

#include <math.h>

#define SMOKE_M1 0x85EBCA6Bu
#define SMOKE_M2 0xC2B2AE35u
#define SMOKE_M3 0x9E3779B1u

static uint32_t smoke_hash_u32(int64_t ix, int64_t iy, uint32_t seed) {
  uint32_t h = (uint32_t)ix * SMOKE_M1;
  h ^= (uint32_t)iy * SMOKE_M2;
  h ^= seed * SMOKE_M3;
  h ^= h >> 15;
  h *= SMOKE_M1;
  h ^= h >> 13;
  h *= SMOKE_M2;
  h ^= h >> 16;
  return h;
}

/* noise.hash_lattice: uint32 hash -> float32 in [0,1), via a float64 divide
 * then a cast down (matches
 * `(h.astype(np.float64)/4294967296.0).astype(f32)`). */
static float smoke_hash_f(int64_t ix, int64_t iy, uint32_t seed) {
  uint32_t h = smoke_hash_u32(ix, iy, seed);
  double v = (double)h / 4294967296.0;
  return (float)v;
}

/* noise.value2d, entirely float32. */
static float smoke_value2d_f(float x, float y, uint32_t seed) {
  float x0 = floorf(x);
  float y0 = floorf(y);
  float fx = x - x0;
  float fy = y - y0;
  int64_t ix = (int64_t)x0;
  int64_t iy = (int64_t)y0;
  float sx = fx * fx * (3.0f - 2.0f * fx);
  float sy = fy * fy * (3.0f - 2.0f * fy);
  float a = smoke_hash_f(ix, iy, seed);
  float b = smoke_hash_f(ix + 1, iy, seed);
  float c = smoke_hash_f(ix, iy + 1, seed);
  float d = smoke_hash_f(ix + 1, iy + 1, seed);
  float top = a + (b - a) * sx;
  float bottom = c + (d - c) * sx;
  return top + (bottom - top) * sy;
}

/* noise.fbm with octaves=3, lacunarity=2.0, gain=0.5 (smoke.py's fbm_plane
 * call), unrolled: amp/freq are exact powers of two so float32 vs float64
 * scalars make no difference to the values themselves. */
static float smoke_fbm3_f(float x, float y, int64_t base_seed) {
  float v0 = smoke_value2d_f(x, y, (uint32_t)(base_seed + 131 * 0));
  float v1 = smoke_value2d_f(x * 2.0f, y * 2.0f, (uint32_t)(base_seed + 131 * 1));
  float v2 = smoke_value2d_f(x * 4.0f, y * 4.0f, (uint32_t)(base_seed + 131 * 2));
  float total = 0.0f;
  total = total + 1.00f * v0;
  total = total + 0.50f * v1;
  total = total + 0.25f * v2;
  return total / 1.75f;
}

/* prims._box1d, restricted to a single contiguous run: padded cumulative sum
 * in float32 (sequential, not a windowed sum -- matters for bit-exactness),
 * then hi-lo, then divide, written back into `plane` in place. Safe in place
 * because every read is from `padded`/`prefix` (independent copies), never
 * from `plane` again once they are built. `padded` must hold at least
 * `n + 2*r` floats and `prefix` at least `n + 2*r + 1` -- both caller-owned,
 * contents ignored on entry. */
static void smoke_box1d_inplace(float *plane, int n, int r, float *padded,
                                float *prefix) {
  int padded_len = n + 2 * r;
  for (int i = 0; i < padded_len; i++) padded[i] = 0.0f;
  for (int i = 0; i < n; i++) padded[r + i] = plane[i];
  prefix[0] = 0.0f;
  for (int t = 0; t < padded_len; t++) prefix[t + 1] = prefix[t] + padded[t];
  float denom = (float)(2 * r + 1);
  for (int k = 0; k < n; k++) {
    float hi = prefix[k + 2 * r + 1];
    float lo = prefix[k];
    plane[k] = (hi - lo) / denom;
  }
}

/* prims.blur(radius_px=s//2, passes=1) on a (rows x cols) plane, in place:
 * box1d along axis=1 (each row, already contiguous) then axis=0 (each
 * column, gathered into `colbuf` since a column has stride `cols`).
 * `padded`/`prefix` must hold at least `max(rows, cols) + 2*r` and
 * `+ 2*r + 1` floats respectively; `colbuf` at least `rows` floats. */
static void smoke_blur_pass(float *plane, int rows, int cols, int r,
                            float *padded, float *prefix, float *colbuf) {
  for (int y = 0; y < rows; y++) {
    smoke_box1d_inplace(plane + (size_t)y * cols, cols, r, padded, prefix);
  }
  for (int x = 0; x < cols; x++) {
    for (int y = 0; y < rows; y++) colbuf[y] = plane[(size_t)y * cols + x];
    smoke_box1d_inplace(colbuf, rows, r, padded, prefix);
    for (int y = 0; y < rows; y++) plane[(size_t)y * cols + x] = colbuf[y];
  }
}

int32_t warlockc_smoke_blob(
    float *out_rgb, float *out_a, int32_t frame_w, int32_t frame_h,
    int32_t y0, int32_t y1, int32_t x0, int32_t x1, float scale, double px,
    double py, double radius, double rag, int64_t fbm_seed, float dx_f32,
    float dy_f32, float nscale_f32, float alpha_mul, float r_rgb,
    float g_rgb, float b_rgb, float *scratch, size_t scratch_len) {
  int win_h = y1 - y0, win_w = x1 - x0;
  if (win_h <= 0 || win_w <= 0) return 0;

  int s = (int)scale;
  int r = s / 2;
  int w_coarse_full = frame_w / s;
  int h_coarse_full = frame_h / s;
  int cy0 = y0 / s;
  int cy1 = (y1 + s - 1) / s;
  int cx0 = x0 / s;
  int cx1 = (x1 + s - 1) / s;
  int ch = cy1 - cy0, cw = cx1 - cx0;
  int big_h = ch * s, big_w = cw * s;
  int max_dim = (big_h > big_w) ? big_h : big_w;

  float *big = NULL;
  int row_off = 0, col_off = 0;
  if (rag > 0.0) {
    size_t need_coarse = (size_t)ch * (size_t)cw;
    size_t need_big = (size_t)big_h * (size_t)big_w;
    size_t need_padded = (size_t)max_dim + 2 * (size_t)r;
    size_t need_prefix = need_padded + 1;
    size_t need_colbuf = (size_t)big_h;
    size_t needed = need_coarse + need_big + need_padded + need_prefix + need_colbuf;
    if (scratch_len < needed) return -1; /* caller falls back to numpy */

    float *coarse = scratch;
    big = coarse + need_coarse;
    float *padded = big + need_big;
    float *prefix = padded + need_padded;
    float *colbuf = prefix + need_prefix;

    for (int i = 0; i < ch; i++) {
      int gy = cy0 + i;
      float ly = ((float)gy + 0.5f) - (float)h_coarse_full / 2.0f;
      float yy = ly / nscale_f32 + dy_f32;
      for (int j = 0; j < cw; j++) {
        int gx = cx0 + j;
        float lx = ((float)gx + 0.5f) - (float)w_coarse_full / 2.0f;
        float xx = lx / nscale_f32 + dx_f32;
        coarse[(size_t)i * cw + j] = smoke_fbm3_f(xx, yy, fbm_seed);
      }
    }

    for (int i = 0; i < big_h; i++) {
      int ci = i / s;
      for (int j = 0; j < big_w; j++) {
        int cj = j / s;
        big[(size_t)i * big_w + j] = coarse[(size_t)ci * cw + cj];
      }
    }

    smoke_blur_pass(big, big_h, big_w, r, padded, prefix, colbuf);

    row_off = y0 - cy0 * s;
    col_off = x0 - cx0 * s;
  }

  float radius_f32 = (float)radius;
  float half_w = (float)frame_w / 2.0f;
  float half_h = (float)frame_h / 2.0f;

  for (int i = 0; i < win_h; i++) {
    int row = y0 + i;
    float winy = ((float)row + 0.5f - half_h) / scale;
    double dy = (double)winy - py;
    for (int j = 0; j < win_w; j++) {
      int col = x0 + j;
      float winx = ((float)col + 0.5f - half_w) / scale;
      double dx = (double)winx - px;
      double dist = sqrt(dx * dx + dy * dy);
      double d = dist / (double)radius_f32;

      double d2;
      if (rag > 0.0) {
        float nij = big[(size_t)(row_off + i) * big_w + (col_off + j)];
        float factor = 1.0f + (nij - 0.5f) * 1.6f * (float)rag;
        d2 = d * (double)factor;
      } else {
        d2 = d;
      }

      double cov = 1.0 - d2;
      if (cov < 0.0) cov = 0.0;
      if (cov > 1.0) cov = 1.0;
      double cov2 = cov * cov * (double)alpha_mul;

      size_t idx_rgb = ((size_t)row * frame_w + col) * 3;
      size_t idx_a = (size_t)row * frame_w + col;

      double one_minus = 1.0 - cov2;

      /* out_rgb *= (1-cov)[...,None]  -- rounds to float32 here */
      float step1_r = (float)((double)out_rgb[idx_rgb + 0] * one_minus);
      float step1_g = (float)((double)out_rgb[idx_rgb + 1] * one_minus);
      float step1_b = (float)((double)out_rgb[idx_rgb + 2] * one_minus);

      /* out_rgb += rgb * cov[...,None]  -- rounds to float32 again */
      out_rgb[idx_rgb + 0] = (float)((double)step1_r + (double)r_rgb * cov2);
      out_rgb[idx_rgb + 1] = (float)((double)step1_g + (double)g_rgb * cov2);
      out_rgb[idx_rgb + 2] = (float)((double)step1_b + (double)b_rgb * cov2);

      /* out_a += cov - out_a * cov  -- one rounding, RHS built in double */
      double oa = (double)out_a[idx_a];
      double rhs = cov2 - oa * cov2;
      out_a[idx_a] = (float)(oa + rhs);
    }
  }

  return 0;
}
