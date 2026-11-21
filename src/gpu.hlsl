// Every GPU pass of the video path: decode, luma pyramid, optical flow, frame generation and
// the final draw. Each resource has its own register so entry points never overlap.
//
// Frame generation, per pair of pictures A and B:
//  1. Dense optical flow both ways (A->B and B->A), coarse to fine over a luma pyramid that
//     starts at half size (quarter size for 4K: motion needs no more detail, and it's the
//     costliest part). Each level searches 8x8 patches every 4 pixels (candidates from the
//     coarser level, the last pair and zero, then Lucas-Kanade refinement), turns the patches
//     into a per-pixel field weighted by how well each explains the pixel, and takes a 3x3
//     median against outliers.
//  2. Each direction is checked against the other: where they disagree the pixel is hidden in
//     the other picture, and borrows the motion of similar visible pixels nearby. Where motion
//     doesn't explain the change better than standing still (fades, flashes, particles), the
//     field is marked unreliable.
//  3. For each in-between time t, every pixel of A is pushed forward along its motion and every
//     pixel of B backward; where several land on one spot, the one visible in both pictures
//     (then the slower one) wins. A spot reached from only one side shows only that side, which
//     is what keeps edges of moving things clean. Unreliable areas fade to a plain blend.
//
// Continuous coordinates: pixel k spans [k, k+1), its centre is k + 0.5. A vector v at a pixel
// of X says where it is in Y: centre + v.

cbuffer Params : register(b0) {
    int4 sz;     // xy: size of the level or frame worked on; zw: second size
    int4 gr;     // xy: patch grid size; zw: size of the next coarser level
    float4 kc;   // YCbCr -> RGB: Cr->R, Cb->G, Cr->G, Cb->B
    float4 misc; // x: t (generation) ; y: SDR white in nits (decode), else full-size pixels per
                 // level-0 pixel ; z: luma offset ; w: luma range
    uint4 fl;    // x: flags ; y: input layout ; z: chroma range
    float4 rect; // present: video rect x, y, w, h in back-buffer pixels
};

#define F_HDR 1
#define F_GAMUT 2
#define F_TEMPORAL 4
#define F_BLACK 8
#define F_SEED 16
#define F_COUNT 32

#define L_NV12 0
#define L_YUY2 1
#define L_RGB32 2

// Inputs
Texture2D<float>  InY       : register(t0);
Texture2D<float2> InUV      : register(t1);
Texture2D<float4> InP       : register(t2);
Texture2D<float4> InFrame   : register(t3);
Texture2D<float>  InLuma    : register(t4);
Texture2D<float>  LX        : register(t5);  // flow: the picture motion is measured from
Texture2D<float>  LY        : register(t6);  // flow: the picture it's measured to
Texture2D<float2> Coarse    : register(t7);  // flow at the next coarser level
Texture2D<float2> Prev      : register(t8);  // last pair's flow at this level
Texture2D<float4> PatchIn   : register(t9);  // per patch: vector, cost, detail
Texture2D<float2> FlowIn    : register(t10);
Texture2D<float2> FlowOther : register(t11); // the opposite direction
Texture2D<float4> FinA      : register(t12); // A->B: vector, mismatch with B->A, unreliability
Texture2D<float4> FinB      : register(t13); // B->A, likewise
Texture2D<uint>   KeyA      : register(t14);
Texture2D<uint>   KeyB      : register(t15);
Texture2D<float4> FA        : register(t16);
Texture2D<float4> FB        : register(t17);
Texture2D<uint>   CountsIn  : register(t18); // per pair, A->B: 0 patches missed, 1 textured patches,
                                             // 2, 3 their cost and detail (the cut test), 4 level-0
                                             // pixels that changed, 5 of those motion can't follow
Texture2D<float4> Shown     : register(t19);
Texture2D<float4> InfoIn    : register(t20);
Texture2D<float4> FX        : register(t21); // the picture motion is measured from, in colour
SamplerState Lin : register(s0);

// Outputs
RWTexture2D<unorm float4> OutFrame  : register(u0);
RWTexture2D<float>        OutLuma   : register(u1);
RWTexture2D<float4>       Out4      : register(u2);
RWTexture2D<float2>       Out2      : register(u3);
RWTexture2D<uint>         Counts    : register(u4);
RWTexture2D<uint>         OutKeyA   : register(u5);
RWTexture2D<uint>         OutKeyB   : register(u6);
RWTexture2D<unorm float4> OutInterp : register(u7);

// ---------------------------------------------------------------------------------------------
// Decode: YCbCr (or RGB) to display-ready sRGB, the same maths as convert.rs.

static const float3x3 BT2020_TO_709 = {
     1.660491, -0.587641, -0.072850,
    -0.124550,  1.132900, -0.008349,
    -0.018151, -0.100579,  1.118730,
};

float pq_to_nits(float e) {
    const float m1 = 2610.0 / 16384.0;
    const float m2 = 2523.0 / 4096.0 * 128.0;
    const float c1 = 3424.0 / 4096.0;
    const float c2 = 2413.0 / 4096.0 * 32.0;
    const float c3 = 2392.0 / 4096.0 * 32.0;
    float p = pow(saturate(e), 1.0 / m2);
    return 10000.0 * pow(max(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1);
}

float srgb_encode(float l) {
    return l <= 0.0031308 ? 12.92 * l : 1.055 * pow(l, 1.0 / 2.4) - 0.055;
}

// Highlight roll-off: linear to the knee, then approaches 1 with a matching slope.
float tonemap(float x) {
    const float knee = 0.9;
    if (x <= knee) return x;
    float t = (x - knee) / (1.0 - knee);
    return knee + (1.0 - knee) * t / (1.0 + t);
}

[numthreads(8, 8, 1)]
void decode(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= (uint2)sz.xy)) return;
    int2 p = int2(id.xy);
    if (fl.y == L_RGB32) {
        OutFrame[p] = float4(InP.Load(int3(p, 0)).rgb, 1.0);
        return;
    }
    float y8, u8, v8;
    if (fl.y == L_NV12) {
        y8 = round(InY.Load(int3(p, 0)) * 255.0);
        float2 uv = InUV.Load(int3(p / 2, 0));
        u8 = round(uv.x * 255.0);
        v8 = round(uv.y * 255.0);
    } else {
        // YUY2 as RGBA texels: Y0 U Y1 V.
        float4 q = InP.Load(int3(p.x / 2, p.y, 0));
        y8 = round(((p.x & 1) ? q.b : q.r) * 255.0);
        u8 = round(q.g * 255.0);
        v8 = round(q.a * 255.0);
    }
    float yy = (y8 - misc.z) / misc.w;
    float cb = (u8 - 128.0) / (float)fl.z;
    float cr = (v8 - 128.0) / (float)fl.z;
    float3 c = float3(yy + kc.x * cr, yy + kc.y * cb + kc.z * cr, yy + kc.w * cb);
    if (fl.x & F_HDR) {
        float3 l = float3(pq_to_nits(c.r), pq_to_nits(c.g), pq_to_nits(c.b)) / misc.y;
        if (fl.x & F_GAMUT) l = max(mul(BT2020_TO_709, l), 0.0);
        // A touch more colour, around luminance in linear light (see convert.rs).
        float lum = dot(l, float3(0.2126, 0.7152, 0.0722));
        l = max(lum + (l - lum) * 1.05, 0.0);
        float m = max(l.r, max(l.g, l.b));
        if (m > 0.9) l *= tonemap(m) / m;
        c = float3(srgb_encode(saturate(l.r)), srgb_encode(saturate(l.g)), srgb_encode(saturate(l.b)));
    }
    OutFrame[p] = float4(saturate(c), 1.0);
}

// ---------------------------------------------------------------------------------------------
// Luma pyramid (0..255 floats): level 0 is 1/misc.y size (2 or 4), each next level half again.

float luma(float4 c) {
    return dot(saturate(c.rgb), float3(77.0, 150.0, 29.0) / 256.0) * 255.0;
}

[numthreads(8, 8, 1)]
void luma0(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= (uint2)sz.xy)) return;
    int s = (int)misc.y;
    int2 p = int2(id.xy) * s;
    int2 m = sz.zw - 1;
    float acc = 0.0;
    [loop] for (int j = 0; j < s; j++)
        [loop] for (int i = 0; i < s; i++) acc += luma(InFrame.Load(int3(min(p + int2(i, j), m), 0)));
    OutLuma[id.xy] = acc / (float)(s * s);
}

[numthreads(8, 8, 1)]
void halve(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= (uint2)sz.xy)) return;
    int2 p = int2(id.xy) * 2;
    int2 m = sz.zw - 1;
    OutLuma[id.xy] = 0.25 * (InLuma.Load(int3(min(p, m), 0)) + InLuma.Load(int3(min(p + int2(1, 0), m), 0))
                           + InLuma.Load(int3(min(p + int2(0, 1), m), 0)) + InLuma.Load(int3(min(p + 1, m), 0)));
}

// ---------------------------------------------------------------------------------------------
// Optical flow, one level, X -> Y. Patches are 8x8 pixels every 4: patch i covers
// [4i - 2, 4i + 6) on each axis, so every pixel is in two patches per axis.

#define FULL 10        // coarsest level: every whole vector within +-FULL
// Lucas-Kanade steps per patch come in fl.z (fewer at the finest level, which starts close).
#define LK_REG 64.0    // keeps flat patches from wandering (gradient^2 units, over the patch)
#define LAMBDA 0.25    // candidate choice: cost per pixel of straying from the coarser level

float lx(int2 p) {
    return LX.Load(int3(clamp(p, int2(0, 0), sz.xy - 1), 0));
}

float ly(int2 p) {
    return LY.Load(int3(clamp(p, int2(0, 0), sz.xy - 1), 0));
}

// Y at continuous position q, bilinear.
float sy(float2 q) {
    return LY.SampleLevel(Lin, q / float2(sz.xy), 0);
}

float2 gradx(int2 p) {
    return 0.5 * float2(lx(p + int2(1, 0)) - lx(p - int2(1, 0)), lx(p + int2(0, 1)) - lx(p - int2(0, 1)));
}

// Sums of v over the group's 64 threads, returned to all of them: 8 threads add up a row each,
// then every thread adds the 8 rows. Two sets of buffers alternate, so a call can start while
// slow threads are still reading the previous call's rows (the barriers of the call in between
// guarantee they're done with the one before).
groupshared float4 gRed[128];
groupshared float4 gRow[16];

float4 group_sum(uint gi, float4 v, inout uint buf) {
    uint base = buf * 64;
    uint rows = buf * 8;
    gRed[base + gi] = v;
    GroupMemoryBarrierWithGroupSync();
    if (gi < 8) {
        float4 s = 0.0;
        [unroll] for (uint k = 0; k < 8; k++) s += gRed[base + gi * 8 + k];
        gRow[rows + gi] = s;
    }
    GroupMemoryBarrierWithGroupSync();
    float4 r = 0.0;
    [unroll] for (uint j = 0; j < 8; j++) r += gRow[rows + j];
    buf ^= 1;
    return r;
}

// Coarsest level: every whole vector within +-FULL, one thread each; the lowest cost (then the
// earliest in scan order) wins. A tiny cost per pixel of length favours standing still on ties.
groupshared uint gBest;

[numthreads(2 * FULL + 1, 2 * FULL + 1, 1)]
void flow_full(uint3 gid : SV_GroupID, uint3 gt : SV_GroupThreadID, uint gi : SV_GroupIndex) {
    if (gi == 0) gBest = 0xFFFFFFFF;
    GroupMemoryBarrierWithGroupSync();
    int2 o = int2(gid.xy) * 4 - 2;
    int2 v = int2(gt.xy) - FULL;
    float sx = 0.0;
    float syy = 0.0;
    [loop] for (int i = 0; i < 64; i++) {
        int2 p = o + int2(i & 7, i >> 3);
        sx += lx(p);
        syy += ly(p + v);
    }
    float dm = (syy - sx) / 64.0;
    float s = 0.0;
    [loop] for (int k = 0; k < 64; k++) {
        int2 p = o + int2(k & 7, k >> 3);
        s += abs(ly(p + v) - lx(p) - dm);
    }
    uint c = (uint)(s * 4.0) + 2 * (uint)(abs(v.x) + abs(v.y));
    InterlockedMin(gBest, (c << 9) | gi);
    GroupMemoryBarrierWithGroupSync();
    if ((gBest & 511) == gi) Out4[gid.xy] = float4(float2(v), 0.0, 0.0);
}

// The coarser level's flow at continuous position q of this level, in this level's pixels.
float2 coarse_at(float2 q) {
    return Coarse.SampleLevel(Lin, q / (2.0 * float2(gr.zw)), 0) * 2.0;
}

// Every level, one 8x8 group per patch (a thread per pixel): the patch's start vector (the
// coarsest level's search, or the best of the coarser level's flow here and around, last
// pair's flow and zero), refined by inverse-compositional Lucas-Kanade. Costs are
// mean-normalised (a uniform brightness change doesn't count). Out: vector, cost (mean
// absolute difference left) and detail (mean absolute deviation of the patch).
[numthreads(8, 8, 1)]
void flow_patch(uint3 gid : SV_GroupID, uint3 gt : SV_GroupThreadID, uint gi : SV_GroupIndex) {
    uint buf = 0;
    int2 pi = int2(gid.xy);
    int2 o = pi * 4 - 2;
    int2 p = o + int2(gt.xy);
    float2 pc = float2(p) + 0.5;
    float tx = lx(p);
    float2 g = gradx(p);
    float2 v0;
    if (fl.x & F_SEED) {
        v0 = PatchIn.Load(int3(pi, 0)).xy;
    } else {
        float2 centre = float2(o) + 4.0;
        float2 base = coarse_at(centre);
        float2 c[8];
        c[0] = base;
        c[1] = coarse_at(centre + float2(-8.0, -8.0));
        c[2] = coarse_at(centre + float2(8.0, -8.0));
        c[3] = coarse_at(centre + float2(-8.0, 8.0));
        c[4] = coarse_at(centre + float2(8.0, 8.0));
        c[5] = float2(0.0, 0.0);
        c[6] = (fl.x & F_TEMPORAL) ? Prev.SampleLevel(Lin, centre / float2(sz.xy), 0) : base;
        c[7] = base;
        float4 d0 = float4(sy(pc + c[0]), sy(pc + c[1]), sy(pc + c[2]), sy(pc + c[3])) - tx;
        float4 d1 = float4(sy(pc + c[4]), sy(pc + c[5]), sy(pc + c[6]), sy(pc + c[7])) - tx;
        float4 m0 = group_sum(gi, d0, buf) / 64.0;
        float4 m1 = group_sum(gi, d1, buf) / 64.0;
        float4 s0 = group_sum(gi, abs(d0 - m0), buf) / 64.0;
        float4 s1 = group_sum(gi, abs(d1 - m1), buf) / 64.0;
        float costs[8] = { s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w };
        float best = 1e30;
        v0 = base;
        [unroll] for (int k = 0; k < 8; k++) {
            float2 dd = abs(c[k] - base);
            float cost = costs[k] + LAMBDA * (dd.x + dd.y);
            if (cost < best) {
                best = cost;
                v0 = c[k];
            }
        }
    }
    float4 hs = group_sum(gi, float4(g.x * g.x, g.x * g.y, g.y * g.y, 0.0), buf);
    float2 sg = group_sum(gi, float4(g, 0.0, 0.0), buf).xy;
    float h11 = hs.x + LK_REG;
    float h12 = hs.y;
    float h22 = hs.z + LK_REG;
    float det = h11 * h22 - h12 * h12;
    float2 v = v0;
    [loop] for (int it = 0; it < (int)fl.z; it++) {
        float d = sy(pc + v) - tx;
        float4 r = group_sum(gi, float4(g * d, d, 0.0), buf);
        float2 b = r.xy - sg * (r.z / 64.0);
        v -= float2(h22 * b.x - h12 * b.y, h11 * b.y - h12 * b.x) / det;
    }
    // Ran off: keep the start.
    if (any(abs(v - v0) > 8.0) || any(isnan(v))) v = v0;
    float d = sy(pc + v) - tx;
    float2 mean = group_sum(gi, float4(d, tx, 0.0, 0.0), buf).xy / 64.0;
    float2 dev = group_sum(gi, float4(abs(d - mean.x), abs(tx - mean.y), 0.0, 0.0), buf).xy / 64.0;
    if (gi == 0 && all(pi < gr.xy)) {
        float cost = dev.x;
        float detail = dev.y;
        Out4[pi] = float4(v, cost, detail);
        if ((fl.x & F_COUNT) && detail >= 3.0) {
            uint old;
            InterlockedAdd(Counts[int2(1, 0)], 1, old);
            if (cost > 0.5 * detail) InterlockedAdd(Counts[int2(0, 0)], 1, old);
            InterlockedAdd(Counts[int2(2, 0)], (uint)(cost * 16.0), old);
            InterlockedAdd(Counts[int2(3, 0)], (uint)(detail * 16.0), old);
        }
    }
}

// Per pixel: the covering patches' vectors, each weighted by how well it explains this pixel.
[numthreads(8, 8, 1)]
void flow_dense(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= (uint2)sz.xy)) return;
    int2 x = int2(id.xy);
    float tx = lx(x);
    int2 k = (x + 2) >> 2;
    float2 acc = 0.0;
    float ws = 0.0;
    [unroll] for (int j = 0; j < 4; j++) {
        int2 pi = clamp(k - 1 + int2(j & 1, j >> 1), int2(0, 0), gr.xy - 1);
        float2 v = PatchIn.Load(int3(pi, 0)).xy;
        float d = abs(sy(float2(x) + 0.5 + v) - tx);
        float w = 1.0 / max(1.0, d);
        acc += w * v;
        ws += w;
    }
    Out2[x] = acc / ws;
}

// Median of 9 by a fixed network of 19 compare-exchanges (no branches, all in registers).
#define CX(a, b) { float lo = min(v[a], v[b]); v[b] = max(v[a], v[b]); v[a] = lo; }

float median9(float v[9]) {
    CX(1, 2) CX(4, 5) CX(7, 8) CX(0, 1) CX(3, 4) CX(6, 7) CX(1, 2) CX(4, 5) CX(7, 8)
    CX(0, 3) CX(5, 8) CX(4, 7) CX(3, 6) CX(1, 4) CX(2, 5) CX(4, 7) CX(4, 2) CX(6, 4) CX(4, 2)
    return v[4];
}

[numthreads(8, 8, 1)]
void flow_smooth(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= (uint2)sz.xy)) return;
    float xs[9];
    float ys[9];
    [unroll] for (int k = 0; k < 9; k++) {
        int2 p = clamp(int2(id.xy) + int2(k % 3 - 1, k / 3 - 1), int2(0, 0), sz.xy - 1);
        float2 v = FlowIn.Load(int3(p, 0));
        xs[k] = v.x;
        ys[k] = v.y;
    }
    Out2[id.xy] = float2(median9(xs), median9(ys));
}

// Level 0, after the median: patches are 8 pixels, so motion bleeds a few pixels across the
// edges of moving things (and of static overlays on moving scenery). Each pixel tries the
// motion found 3-4 pixels away in 8 directions and keeps whichever explains its own 3x3 best,
// which puts the motion edge on the picture's edge. A small cost for leaving its own vector
// keeps flat areas, where everything matches, as they are.
#define SNAP_LAMBDA 0.5

static const float2 SNAP_DIRS[8] = {
    float2(4, 0), float2(-4, 0), float2(0, 4), float2(0, -4),
    float2(3, 3), float2(-3, 3), float2(3, -3), float2(-3, -3),
};

float snap_cost(int2 x, float2 v) {
    float s = 0.0;
    [unroll] for (int k = 0; k < 9; k++) {
        int2 n = x + int2(k % 3 - 1, k / 3 - 1);
        s += abs(sy(float2(n) + 0.5 + v) - lx(n));
    }
    return s / 9.0;
}

[numthreads(8, 8, 1)]
void flow_snap(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= (uint2)sz.xy)) return;
    int2 x = int2(id.xy);
    int2 m = sz.xy - 1;
    float2 own = FlowIn.Load(int3(x, 0));
    float2 best_v = own;
    float best = snap_cost(x, own);
    [loop] for (int d = 0; d < 8; d++) {
        int2 o = int2(SNAP_DIRS[d]);
        float2 v = FlowIn.Load(int3(clamp(x + o, int2(0, 0), m), 0));
        float2 dv = abs(v - own);
        float c = snap_cost(x, v) + SNAP_LAMBDA * (dv.x + dv.y);
        if (c < best) {
            best = c;
            best_v = v;
        }
    }
    Out2[x] = best_v;
}

// ---------------------------------------------------------------------------------------------
// Checks at level 0, per direction X -> Y.

// A pixel counts as hidden in the other picture when the opposite direction doesn't bring it
// back to within OCC_ABS level-0 pixels plus OCC_REL of its motion (fast motion is measured
// less exactly).
#define OCC_ABS 0.75
#define OCC_REL 0.1
// The mismatch is stored relative to that allowance: above 1 is hidden. Pixels moving out of
// the picture get LEAVING: as hidden as can be when competing for a spot, but not filled.
#define OCCLUDED 1.0
#define LEAVING -100.0
// Levels of difference per pixel motion may leave unexplained before the area counts as
// unreliable at all.
#define UNEXPLAINED 1.5
// Motion leaving more than this share of the change that standing still leaves (ramping to
// fully unreliable over the next RATIO_RAMP) isn't motion: a menu highlight jumping, text
// changing, a dissolve. Following it scatters pieces of the old picture.
#define RATIO_START 0.4
#define RATIO_RAMP 0.3
// Hidden pixels come in bands beside moving edges. Where more than HIDDEN_START of a 31x31
// window (level-0 pixels, sampled 5 apart) is hidden, ramping over HIDDEN_RAMP, nothing
// matched at all (a screen changing in place), so blend.
#define HIDDEN_STEP 5
#define HIDDEN_START 0.7
#define HIDDEN_RAMP 0.2
// Levels of difference between A and B (unmoved) for a level-0 pixel to count as changed.
#define CHANGED 4.0
// The UI-change test: when more than LOST_START of what changed can't be followed (ramping to
// all of the frame over LOST_RAMP), the change isn't motion but a menu, a highlight jumping,
// a screen opening. Following parts of it tears it (pieces move, the rest blends), so the whole
// frame cross-fades. Gameplay stays under 0.35 on the test clips, menu changes are 0.65 and up.
#define LOST_START 0.45
#define LOST_RAMP 0.15

// x: mismatch with the opposite flow (relative, see above); y: |Y moved - X|; z: |Y unmoved - X|.
[numthreads(8, 8, 1)]
void flow_check(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= (uint2)sz.xy)) return;
    int2 x = int2(id.xy);
    float2 v = FlowIn.Load(int3(x, 0));
    float2 q = float2(x) + 0.5 + v;
    float2 size = float2(sz.xy);
    float tx = lx(x);
    float2 w = FlowOther.SampleLevel(Lin, q / size, 0);
    float e = length(v + w) / (OCC_ABS + OCC_REL * 0.5 * (length(v) + length(w)));
    float moved = abs(sy(q) - tx);
    // Leaving the picture: hidden in Y, but the vector is fine (nothing to check it against),
    // so it's marked LEAVING to be kept rather than filled.
    if (any(q < 0.0) || any(q > size)) {
        e = LEAVING;
        moved = 0.0;
    }
    Out4[x] = float4(e, moved, abs(ly(x) - tx), 0.0);
}

static const float2 DIRS[8] = {
    float2(1, 0), float2(-1, 0), float2(0, 1), float2(0, -1),
    float2(1, 1), float2(-1, 1), float2(1, -1), float2(-1, -1),
};

// What the full-size pixels around level-0 pixel p look like (4x4 samples, spaced to cover two
// level-0 pixels each way): mean colour, and how busy they are (mean absolute deviation of
// their luma), all in levels.
float4 look(int2 p) {
    int step = (int)misc.y / 2;
    int2 o = p * (int)misc.y - step;
    int2 m = sz.zw - 1;
    float3 c[16];
    float3 s = 0.0;
    [unroll] for (int k = 0; k < 16; k++) {
        c[k] = FX.Load(int3(clamp(o + int2(k & 3, k >> 2) * step, int2(0, 0), m), 0)).rgb;
        s += c[k];
    }
    s /= 16.0;
    float ml = dot(s, float3(0.299, 0.587, 0.114));
    float busy = 0.0;
    [unroll] for (int j = 0; j < 16; j++) busy += abs(dot(c[j], float3(0.299, 0.587, 0.114)) - ml);
    return float4(s, busy / 16.0) * 255.0;
}

// Hidden pixels take the motion of visible pixels nearby (their own vector matched nothing):
// look outward in 8 directions for the first visible pixel (hidden bands beside fast motion
// are wide) and take the motion of the side that looks most alike (colour, and how busy:
// a HUD's dark bits can match dark scenery in colour alone), which is the layer this pixel
// belongs to.
[numthreads(8, 8, 1)]
void flow_fill(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= (uint2)sz.xy)) return;
    int2 x = int2(id.xy);
    float2 v = FlowIn.Load(int3(x, 0));
    int2 m = sz.xy - 1;
    if (InfoIn.Load(int3(x, 0)).x > OCCLUDED) {
        float4 cx = look(x);
        float best = 1e30;
        [loop] for (int d = 0; d < 8; d++) {
            int2 dir = int2(DIRS[d]);
            [loop] for (int s = 1; s <= 64; s *= 2) {
                int2 n = x + dir * s;
                if (any(n < 0) || any(n > m)) break;
                if (InfoIn.Load(int3(n, 0)).x <= OCCLUDED) {
                    float4 dc = abs(look(n) - cx);
                    float c = dc.r + dc.g + dc.b + 3.0 * dc.a + 0.1 * (float)s;
                    if (c < best) {
                        best = c;
                        v = FlowIn.Load(int3(n, 0));
                    }
                    break;
                }
            }
        }
    }
    Out2[x] = v;
}

// Hidden pixels: the median of the filled motion around them (3x3 samples 3 apart), so a hidden area goes with
// one side as a whole instead of pixel by pixel. Unreliability: around each pixel, how much of
// the change motion leaves unexplained compared with standing still, and whether the area is
// mostly hidden (no match at all).
[numthreads(8, 8, 1)]
void flow_final(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= (uint2)sz.xy)) return;
    int2 x = int2(id.xy);
    int2 m = sz.xy - 1;
    float4 info = InfoIn.Load(int3(x, 0));
    float2 v = FlowIn.Load(int3(x, 0));
    if (info.x > OCCLUDED) {
        float xs[9];
        float ys[9];
        [unroll] for (int k = 0; k < 9; k++) {
            float2 n = FlowIn.Load(int3(clamp(x + int2(k % 3 - 1, k / 3 - 1) * 3, int2(0, 0), m), 0));
            xs[k] = n.x;
            ys[k] = n.y;
        }
        v = float2(median9(xs), median9(ys));
    }
    // Over a 12x12 window (bilinear taps between pixels average 2x2 each), hidden pixels
    // included: a thin band of them beside a moving edge barely moves the average, while an
    // area where nothing matches (a fade, something appearing, a blur of fast motion) does.
    float2 acc = 0.0;
    [loop] for (int k2 = 0; k2 < 36; k2++) {
        float2 q = float2(x) + 0.5 + float2(k2 % 6 - 2.5, k2 / 6 - 2.5) * 2.0;
        acc += InfoIn.SampleLevel(Lin, q / float2(sz.xy), 0).yz;
    }
    float smc = acc.x / 36.0;
    float s0 = acc.y / 36.0;
    float u = saturate((smc - 0.6 * s0 - UNEXPLAINED) / (0.3 * s0 + 3.0));
    u = max(u, saturate((smc / max(s0, 1e-3) - RATIO_START) / RATIO_RAMP) * saturate((smc - UNEXPLAINED) / 2.0));
    float hidden = 0.0;
    [loop] for (int k3 = 0; k3 < 49; k3++) {
        int2 n = clamp(x + int2(k3 % 7 - 3, k3 / 7 - 3) * HIDDEN_STEP, int2(0, 0), m);
        hidden += InfoIn.Load(int3(n, 0)).x > OCCLUDED ? 1.0 : 0.0;
    }
    u = max(u, saturate((hidden / 49.0 - HIDDEN_START) / HIDDEN_RAMP));
    if ((fl.x & F_COUNT) && info.z > CHANGED) {
        uint old;
        InterlockedAdd(Counts[int2(4, 0)], 1, old);
        if (u > 0.5 || info.x > OCCLUDED) InterlockedAdd(Counts[int2(5, 0)], 1, old);
    }
    Out4[x] = float4(v, info.x, u);
}

// ---------------------------------------------------------------------------------------------
// Generation at time t, full size.

#define EMPTY 0xFFFFFFFF
#define INDEX_BITS 23

// Push every pixel of A (z = 0) forward by t, or of B (z = 1) back by 1 - t, onto the 2x2
// pixels around where it lands. Lowest key wins a pixel: visible in both pictures first, then
// slower motion (static overlays over moving scenery), then any.
[numthreads(16, 16, 1)]
void project(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= (uint2)sz.xy)) return;
    float t = misc.x;
    float2 size = float2(sz.xy);
    float2 xc = float2(id.xy) + 0.5;
    float4 f = id.z == 0 ? FinA.SampleLevel(Lin, xc / size, 0) : FinB.SampleLevel(Lin, xc / size, 0);
    float2 v = f.xy * misc.y;
    float2 yc = xc + (id.z == 0 ? t : 1.0 - t) * v;
    int2 base = (int2)floor(yc - 0.5);
    uint vis = f.z < 0.0 ? 3 : (f.z < 0.5 ? 0 : (f.z < 1.0 ? 1 : (f.z < 2.0 ? 2 : 3)));
    uint speed = min((uint)(length(v) * 0.25), 127u);
    uint key = (((vis << 7) | speed) << INDEX_BITS) | (id.y * (uint)sz.x + id.x);
    [unroll] for (int k = 0; k < 4; k++) {
        int2 p = base + int2(k & 1, k >> 1);
        if (any(p < 0) || any(p >= sz.xy)) continue;
        if (id.z == 0) InterlockedMin(OutKeyA[p], key);
        else InterlockedMin(OutKeyB[p], key);
    }
}

// One side at output pixel p: how much of the 3x3 around it was reached (0..1), and the vector
// (A -> B convention) and unreliability of what landed on p (or the best neighbour).
float side(int2 p, bool fromA, out float2 v, out float u) {
    uint best = EMPTY;
    float hits = 0.0;
    [unroll] for (int k = 0; k < 9; k++) {
        int2 n = clamp(p + int2(k % 3 - 1, k / 3 - 1), int2(0, 0), sz.xy - 1);
        uint key = fromA ? KeyA.Load(int3(n, 0)) : KeyB.Load(int3(n, 0));
        if (key != EMPTY) {
            hits += 1.0;
            best = min(best, key);
        }
    }
    // The centre's own winner comes first; a neighbour's only if nothing landed here.
    uint centre = fromA ? KeyA.Load(int3(p, 0)) : KeyB.Load(int3(p, 0));
    if (centre != EMPTY) best = centre;
    v = 0.0;
    u = 0.0;
    if (best == EMPTY) return 0.0;
    uint idx = best & ((1u << INDEX_BITS) - 1);
    float2 xc = float2(idx % (uint)sz.x, idx / (uint)sz.x) + 0.5;
    float4 f = fromA ? FinA.SampleLevel(Lin, xc / float2(sz.xy), 0) : FinB.SampleLevel(Lin, xc / float2(sz.xy), 0);
    v = fromA ? f.xy * misc.y : -f.xy * misc.y;
    u = f.w;
    return hits / 9.0;
}

[numthreads(16, 16, 1)]
void compose(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= (uint2)sz.xy)) return;
    int2 p = int2(id.xy);
    // A cut: nothing to follow, so hold A as the source itself does until B is due (the cut then
    // lands on time, in step with the sound).
    uint cost = CountsIn.Load(int3(2, 0, 0));
    uint detail = CountsIn.Load(int3(3, 0, 0));
    // Unrelated pictures score about 1 (0.97 and up on real cuts); hard motion up to 0.83 (fast
    // camera turns over dark foliage, at 15 fps).
    if (detail > 0 && (float)cost > 0.9 * (float)detail) {
        OutInterp[p] = FA.Load(int3(p, 0));
        return;
    }
    float t = misc.x;
    float2 size = float2(sz.xy);
    float2 c = float2(p) + 0.5;
    uint changed = CountsIn.Load(int3(4, 0, 0));
    uint lost = CountsIn.Load(int3(5, 0, 0));
    float ui = changed > 0 ? saturate(((float)lost / (float)changed - LOST_START) / LOST_RAMP) : 0.0;
    if (ui >= 1.0) {
        OutInterp[p] = float4(lerp(FA.Load(int3(p, 0)).rgb, FB.Load(int3(p, 0)).rgb, t), 1.0);
        return;
    }
    float2 va, vb;
    float ua, ub;
    float visa = side(p, true, va, ua);
    float visb = side(p, false, vb, ub);
    float wa = (1.0 - t) * visa;
    float wb = t * visb;
    if (wa + wb < 1e-3) {
        // Reached from neither side: stand still and blend.
        wa = 1.0 - t;
        wb = t;
        va = 0.0;
        vb = 0.0;
        ua = 1.0;
        ub = 1.0;
    }
    float3 a = FA.SampleLevel(Lin, (c - t * va) / size, 0).rgb;
    float3 b = FB.SampleLevel(Lin, (c + (1.0 - t) * vb) / size, 0).rgb;
    float3 moved = (wa * a + wb * b) / (wa + wb);
    float u = (wa * ua + wb * ub) / (wa + wb);
    // Static things over moving scenery (HUD, subtitles, see-through panels) can pick up the
    // scenery's motion. Over the 3x3 around p, compare how well standing still matches A to B
    // with how well the motion does; where still is clearly better, stay still. Where only one
    // side is visible the motion can't be checked, so only near-identical pixels stay.
    // Only where there's detail standing still: text sliding over a flat background also looks
    // unchanged between A and B at the spot it passes through, and must keep moving.
    float d0 = 0.0;
    float dm = 0.0;
    float busy = 0.0;
    float3 ac = FA.Load(int3(p, 0)).rgb;
    [unroll] for (int k = 0; k < 9; k++) {
        float2 o = float2(k % 3 - 1, k / 3 - 1);
        int2 n = clamp(p + int2(o), int2(0, 0), sz.xy - 1);
        float3 an = FA.Load(int3(n, 0)).rgb;
        float3 s = abs(an - FB.Load(int3(n, 0)).rgb);
        d0 += s.r + s.g + s.b;
        float3 m = abs(FA.SampleLevel(Lin, (c + o - t * va) / size, 0).rgb - FB.SampleLevel(Lin, (c + o + (1.0 - t) * vb) / size, 0).rgb);
        dm += m.r + m.g + m.b;
        float3 dd = abs(an - ac);
        busy += dd.r + dd.g + dd.b;
    }
    // Means per channel, in levels.
    d0 *= 255.0 / 27.0;
    dm *= 255.0 / 27.0;
    busy *= 255.0 / 24.0;
    float keep = (visa > 0.5 && visb > 0.5) ? saturate((dm - d0 - 1.0) / (0.5 * d0 + 3.0)) : saturate(1.0 - d0 / 2.0);
    keep *= saturate(busy / 8.0 - 0.25);
    u = max(u, max(keep, ui));
    float3 still = lerp(FA.Load(int3(p, 0)).rgb, FB.Load(int3(p, 0)).rgb, t);
    OutInterp[p] = float4(lerp(moved, still, u), 1.0);
}

// ---------------------------------------------------------------------------------------------
// Present: the video fitted into rect (whole pixels), black around it. 1:1 is copied exactly;
// smaller averages the source footprint; larger is bilinear.

float4 present_vs(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

// Lanczos with two lobes: sharp, with little ringing.
float lanczos2(float x) {
    x = abs(x);
    if (x < 1e-4) return 1.0;
    if (x >= 2.0) return 0.0;
    float px = 3.14159265 * x;
    return 2.0 * sin(px) * sin(px * 0.5) / (px * px);
}

#define MAX_TAPS 18

float4 present_ps(float4 pos : SV_Position) : SV_Target {
    float2 p = pos.xy - rect.xy;
    if ((fl.x & F_BLACK) || any(p < 0.0) || any(p >= rect.zw)) return float4(0.0, 0.0, 0.0, 1.0);
    float2 src = float2(sz.zw);
    float2 scale = src / rect.zw;
    if (all(abs(scale - 1.0) < 1e-4)) return float4(Shown.Load(int3(int2(p), 0)).rgb, 1.0);
    // Lanczos-2 resampling. Shrinking widens the kernel by the scale, so every source pixel
    // counts (no aliasing); growing keeps it at two pixels each way (sharper than bilinear).
    float2 q = p * scale; // this pixel's centre in source pixels
    float2 widen = clamp(scale, 1.0, 4.0);
    float2 radius = 2.0 * widen;
    int2 lo = (int2)floor(q - radius - 0.5) + 1;
    int2 n = min((int2)ceil(2.0 * radius) + 1, MAX_TAPS);
    float wx[MAX_TAPS];
    [loop] for (int i = 0; i < n.x; i++) wx[i] = lanczos2((lo.x + i + 0.5 - q.x) / widen.x);
    float3 acc = 0.0;
    float total = 0.0;
    [loop] for (int j = 0; j < n.y; j++) {
        float wy = lanczos2((lo.y + j + 0.5 - q.y) / widen.y);
        if (wy == 0.0) continue;
        int y = clamp(lo.y + j, 0, sz.w - 1);
        [loop] for (int i = 0; i < n.x; i++) {
            float w = wx[i] * wy;
            acc += w * Shown.Load(int3(clamp(lo.x + i, 0, sz.z - 1), y, 0)).rgb;
            total += w;
        }
    }
    return float4(saturate(acc / total), 1.0);
}
