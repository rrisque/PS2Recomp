#ifndef PS2_RUNTIME_MACROS_H
#define PS2_RUNTIME_MACROS_H
#include <cstdint>
#include <cmath>
#include <cstring>
#include <bit>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(USE_SSE2NEON)
#include "sse2neon.h"
#else
#include <immintrin.h> // For SSE/AVX intrinsics
#endif

#include "ps2_runtime.h"

static inline int32_t Ps2ExtractEpi32(__m128i v, int index)
{
    switch (index & 3)
    {
    case 0:
        return _mm_extract_epi32(v, 0);
    case 1:
        return _mm_extract_epi32(v, 1);
    case 2:
        return _mm_extract_epi32(v, 2);
    default:
        return _mm_extract_epi32(v, 3);
    }
}

static inline int64_t Ps2ExtractEpi64(__m128i v, int index)
{
    if ((index & 1) == 0)
    {
        return _mm_cvtsi128_si64(v);
    }
    else
    {
        return _mm_extract_epi64(v, 1);
    }
}

static inline uint32_t ps2_clz32(uint32_t x)
{
    return static_cast<uint32_t>(std::countl_zero(x));
}

static inline uint64_t Ps2HiLoToU64(uint64_t hi, uint64_t lo)
{
    return ((hi & 0xFFFFFFFFull) << 32) | (lo & 0xFFFFFFFFull);
}

static inline uint64_t Ps2SignExt32ToU64(uint32_t v)
{
    return (uint64_t)(int64_t)(int32_t)v;
}

// PLZCW: Count leading bits that match the sign bit, minus 1.
// For positive values: count leading zeros minus 1 (excludes sign bit).
// For negative values: count leading ones minus 1 (excludes sign bit).
// Special cases: 0x00000000 -> 31, 0xFFFFFFFF -> 31.
static inline uint32_t ps2_plzcw32(uint32_t x)
{
    if (x == 0 || x == 0xFFFFFFFF)
        return 31;
    if (x & 0x80000000u)
        x = ~x; // If sign bit set, invert to count leading ones as zeros
    return static_cast<uint32_t>(std::countl_zero(x)) - 1;
}

#define PS2_BLENDV_PS(a, b, mask) _mm_blendv_ps((a), (b), (mask))
#define PS2_MIN_EPI32(a, b) _mm_min_epi32((a), (b))
#define PS2_MAX_EPI32(a, b) _mm_max_epi32((a), (b))
#define PS2_SHUFFLE_EPI8(v, mask) _mm_shuffle_epi8((v), (mask))

#define PS2_EXTRACT_EPI32(v, i) Ps2ExtractEpi32((v), (i))
#define PS2_EXTRACT_EPI64(v, i) Ps2ExtractEpi64((v), (i))

#define PS2_EXTRACT_EPI32_0(v) Ps2ExtractEpi32((v), 0)
#define PS2_EXTRACT_EPI32_1(v) Ps2ExtractEpi32((v), 1)
#define PS2_EXTRACT_EPI32_2(v) Ps2ExtractEpi32((v), 2)
#define PS2_EXTRACT_EPI32_3(v) Ps2ExtractEpi32((v), 3)

#define PS2_EXTRACT_EPI64_0(v) Ps2ExtractEpi64((v), 0)
#define PS2_EXTRACT_EPI64_1(v) Ps2ExtractEpi64((v), 1)

// Basic MIPS arithmetic operations
#define ADD32(a, b) ((uint32_t)((a) + (b)))
// Overflow-checked 32-bit add/sub (ADD/ADDI/SUB). The sum is formed in unsigned arithmetic:
// signed int32 overflow is UB in C++ and GCC/Clang fold the subsequent overflow test away
// (e.g. `sub rd, $zero, rt` with rt == INT32_MIN never raised the exception).
#define ADD32_OV(rs, rt, result32, overflow)                               \
    do                                                                     \
    {                                                                      \
        const uint32_t _a = (uint32_t)(int32_t)(rs);                       \
        const uint32_t _b = (uint32_t)(int32_t)(rt);                       \
        const uint32_t _r = _a + _b;                                       \
        overflow = ((~(_a ^ _b) & (_a ^ _r)) & 0x80000000u) != 0;          \
        result32 = _r;                                                     \
    } while (0);
#define SUB32(a, b) ((uint32_t)((a) - (b)))
#define SUB32_OV(rs, rt, result32, overflow)                               \
    do                                                                     \
    {                                                                      \
        const uint32_t _a = (uint32_t)(int32_t)(rs);                       \
        const uint32_t _b = (uint32_t)(int32_t)(rt);                       \
        const uint32_t _r = _a - _b;                                       \
        overflow = (((_a ^ _b) & (_a ^ _r)) & 0x80000000u) != 0;           \
        result32 = _r;                                                     \
    } while (0);
#define MUL32(a, b) ((uint32_t)((a) * (b)))
#define DIV32(a, b) ((uint32_t)((a) / (b)))
#define AND32(a, b) ((uint32_t)((a) & (b)))
#define OR32(a, b) ((uint32_t)((a) | (b)))
#define XOR32(a, b) ((uint32_t)((a) ^ (b)))
#define NOR32(a, b) ((uint32_t)(~((a) | (b))))
#define SLL32(a, b) ((uint32_t)((a) << (b)))
#define SRL32(a, b) ((uint32_t)((a) >> (b)))
#define SRA32(a, b) ((uint32_t)((int32_t)(a) >> (b)))
#define SLT32(a, b) ((uint32_t)((int32_t)(a) < (int32_t)(b) ? 1 : 0))
#define SLTU32(a, b) ((uint32_t)((a) < (b) ? 1 : 0))

// PS2-specific 128-bit MMI operations
#define PS2_PEXTLW(a, b) _mm_unpacklo_epi32((__m128i)(b), (__m128i)(a))
#define PS2_PEXTUW(a, b) _mm_unpackhi_epi32((__m128i)(b), (__m128i)(a))
#define PS2_PEXTLH(a, b) _mm_unpacklo_epi16((__m128i)(b), (__m128i)(a))
#define PS2_PEXTUH(a, b) _mm_unpackhi_epi16((__m128i)(b), (__m128i)(a))
#define PS2_PEXTLB(a, b) _mm_unpacklo_epi8((__m128i)(b), (__m128i)(a))
#define PS2_PEXTUB(a, b) _mm_unpackhi_epi8((__m128i)(b), (__m128i)(a))
#define PS2_PADDW(a, b) _mm_add_epi32((__m128i)(a), (__m128i)(b))
#define PS2_PSUBW(a, b) _mm_sub_epi32((__m128i)(a), (__m128i)(b))
#define PS2_PMAXW(a, b) PS2_MAX_EPI32((__m128i)(a), (__m128i)(b))
#define PS2_PMINW(a, b) PS2_MIN_EPI32((__m128i)(a), (__m128i)(b))
#define PS2_PADDH(a, b) _mm_add_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PSUBH(a, b) _mm_sub_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PMAXH(a, b) _mm_max_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PMINH(a, b) _mm_min_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PADDB(a, b) _mm_add_epi8((__m128i)(a), (__m128i)(b))
#define PS2_PSUBB(a, b) _mm_sub_epi8((__m128i)(a), (__m128i)(b))
#define PS2_PAND(a, b) _mm_and_si128((__m128i)(a), (__m128i)(b))
#define PS2_POR(a, b) _mm_or_si128((__m128i)(a), (__m128i)(b))
#define PS2_PXOR(a, b) _mm_xor_si128((__m128i)(a), (__m128i)(b))
#define PS2_PNOR(a, b) _mm_xor_si128(_mm_or_si128((__m128i)(a), (__m128i)(b)), _mm_set1_epi32(0xFFFFFFFF))

// PS2 VU (Vector Unit) / FPU float semantics.
// The PS2 FPU and VUs have no Inf/NaN/denormals. Following the PCSX2 interpreter (fpuDouble/vuDouble on
// operands, checkOverflow/checkUnderflow/VU_MAC_UPDATE on results):
//   exponent 0   (zero/denormal) -> signed zero
//   exponent 255 (Inf/NaN)       -> signed FLT_MAX
// Rounding is the host's round-to-nearest (real hardware truncates; not modelled).
static inline uint32_t Ps2FloatBits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}
static inline float Ps2BitsFloat(uint32_t u)
{
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}
static inline float Ps2FNorm(float v)
{
    const uint32_t u = Ps2FloatBits(v);
    const uint32_t e = u & 0x7F800000u;
    if (e == 0u)
        return Ps2BitsFloat(u & 0x80000000u);
    if (e == 0x7F800000u)
        return Ps2BitsFloat((u & 0x80000000u) | 0x7F7FFFFFu);
    return v;
}
static inline __m128 Ps2VNorm(__m128 v)
{
    const __m128i expMask = _mm_set1_epi32(0x7F800000);
    const __m128i u = _mm_castps_si128(v);
    const __m128i e = _mm_and_si128(u, expMask);
    const __m128i sign = _mm_and_si128(u, _mm_set1_epi32(static_cast<int>(0x80000000u)));
    __m128i r = _mm_blendv_epi8(u, sign, _mm_cmpeq_epi32(e, _mm_setzero_si128()));
    r = _mm_blendv_epi8(r, _mm_or_si128(sign, _mm_set1_epi32(0x7F7FFFFF)), _mm_cmpeq_epi32(e, expMask));
    return _mm_castsi128_ps(r);
}
// Older names, kept for existing callers.
static inline __m128 Ps2VClamp(__m128 v) { return Ps2VNorm(v); }
static inline float Ps2FClamp(float v) { return Ps2FNorm(v); }

// Keeps the compiler from contracting a*b+c into an FMA (-march=x86-64-v3 + -ffp-contract=fast):
// the PS2 multiply-add rounds the product before the add.
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define PS2_FP_BARRIER(x) __asm__("" : "+x"(x))
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__aarch64__)
#define PS2_FP_BARRIER(x) __asm__("" : "+w"(x))
#else
#define PS2_FP_BARRIER(x) ((void)0)
#endif

static inline __m128 Ps2VAdd(__m128 a, __m128 b) { return Ps2VNorm(_mm_add_ps(Ps2VNorm(a), Ps2VNorm(b))); }
static inline __m128 Ps2VSub(__m128 a, __m128 b) { return Ps2VNorm(_mm_sub_ps(Ps2VNorm(a), Ps2VNorm(b))); }
static inline __m128 Ps2VMul(__m128 a, __m128 b) { return Ps2VNorm(_mm_mul_ps(Ps2VNorm(a), Ps2VNorm(b))); }
// VMADD/VMSUB: acc +- fs*ft; the product itself is not saturated (PCSX2 _vuOpMADD).
static inline __m128 Ps2VMadd(__m128 acc, __m128 a, __m128 b)
{
    __m128 p = _mm_mul_ps(Ps2VNorm(a), Ps2VNorm(b));
    PS2_FP_BARRIER(p);
    return Ps2VNorm(_mm_add_ps(Ps2VNorm(acc), p));
}
static inline __m128 Ps2VMsub(__m128 acc, __m128 a, __m128 b)
{
    __m128 p = _mm_mul_ps(Ps2VNorm(a), Ps2VNorm(b));
    PS2_FP_BARRIER(p);
    return Ps2VNorm(_mm_sub_ps(Ps2VNorm(acc), p));
}
// VMAX/VMINI compare the raw bit patterns as sign-magnitude integers (so -0 < +0, no NaN issues).
static inline __m128 Ps2VMax(__m128 a, __m128 b)
{
    const __m128i ia = _mm_castps_si128(a), ib = _mm_castps_si128(b);
    const __m128i bothNeg = _mm_srai_epi32(_mm_and_si128(ia, ib), 31);
    return _mm_castsi128_ps(_mm_blendv_epi8(_mm_max_epi32(ia, ib), _mm_min_epi32(ia, ib), bothNeg));
}
static inline __m128 Ps2VMin(__m128 a, __m128 b)
{
    const __m128i ia = _mm_castps_si128(a), ib = _mm_castps_si128(b);
    const __m128i bothNeg = _mm_srai_epi32(_mm_and_si128(ia, ib), 31);
    return _mm_castsi128_ps(_mm_blendv_epi8(_mm_min_epi32(ia, ib), _mm_max_epi32(ia, ib), bothNeg));
}
// VFTOIn: truncate v*2^n; |v*2^n| >= 2^31 (incl. Inf/NaN patterns) saturates by sign.
static inline __m128 Ps2VFtoi(__m128 v, float scale)
{
    const __m128 f = _mm_mul_ps(v, _mm_set1_ps(scale));
    const __m128i bits = _mm_castps_si128(f);
    const __m128i big = _mm_cmpgt_epi32(_mm_and_si128(bits, _mm_set1_epi32(0x7F800000)), _mm_set1_epi32(0x4EFFFFFF));
    const __m128i sat = _mm_xor_si128(_mm_srai_epi32(bits, 31), _mm_set1_epi32(0x7FFFFFFF));
    return _mm_castsi128_ps(_mm_blendv_epi8(_mm_cvttps_epi32(f), sat, big));
}
// VITOFn: (float)int * 2^-n
static inline __m128 Ps2VItof(__m128 v, float scale)
{
    return _mm_mul_ps(_mm_cvtepi32_ps(_mm_castps_si128(v)), _mm_set1_ps(scale));
}
// VCLIPw.xyz fs, ft: 6 judgement bits (+x,-x,+y,-y,+z,-z) against |ft.w| (integer compare, PCSX2 _vuCLIP).
static inline uint32_t Ps2VClipFlags(__m128 fs, float ftw)
{
    uint32_t w = Ps2FloatBits(ftw);
    w = (w & 0x7F800000u) ? (w & 0x7FFFFFFFu) : 0x007FFFFFu;
    const __m128i lim = _mm_set1_epi32(static_cast<int>(w));
    const __m128i v = _mm_castps_si128(fs);
    const uint32_t pos = static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(v, lim))));
    const uint32_t neg = static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(
        _mm_cmpgt_epi32(_mm_xor_si128(v, _mm_set1_epi32(static_cast<int>(0x80000000u))), lim))));
    return ((pos & 1u) << 0) | ((neg & 1u) << 1) | ((pos & 2u) << 1) | ((neg & 2u) << 2) |
           ((pos & 4u) << 2) | ((neg & 4u) << 3);
}
static inline float Ps2VLane(__m128 v, int lane)
{
    alignas(16) float t[4];
    _mm_store_ps(t, v);
    return t[lane & 3];
}

// Division (FPU div.s, VU0 VDIV): divisor with exponent 0 -> +-FLT_MAX (xor of the raw signs), even 0/0.
static inline float Ps2FDiv(float a, float b)
{
    const uint32_t ub = Ps2FloatBits(b);
    if ((ub & 0x7F800000u) == 0u)
        return Ps2BitsFloat(((Ps2FloatBits(a) ^ ub) & 0x80000000u) | 0x7F7FFFFFu);
    return Ps2FNorm(Ps2FNorm(a) / Ps2FNorm(b));
}
// Divide-by-zero diagnostics (PS2X_TRACE_DIVZERO): records the guest PC of x/0 in FPU/VU0 code.
void Ps2NoteDivZero(uint32_t pc);
static inline float Ps2FDivAt(uint32_t pc, float a, float b)
{
    if ((Ps2FloatBits(b) & 0x7F800000u) == 0u)
        Ps2NoteDivZero(pc);
    return Ps2FDiv(a, b);
}
// VU0 VSQRT: sqrt(|ft|)
static inline float Ps2VSqrt(float t)
{
    return Ps2FNorm(sqrtf(fabsf(Ps2FNorm(t))));
}
// VU0 VRSQRT: fs/sqrt(|ft|); ft==0 -> +-FLT_MAX, or +-0 when fs==0 too (sign = xor of the raw signs).
static inline float Ps2VRsqrt(float s, float t)
{
    const uint32_t us = Ps2FloatBits(s), ut = Ps2FloatBits(t);
    if ((ut & 0x7F800000u) == 0u)
    {
        const uint32_t sign = (us ^ ut) & 0x80000000u;
        return Ps2BitsFloat((us & 0x7F800000u) ? (sign | 0x7F7FFFFFu) : sign);
    }
    return Ps2FNorm(Ps2FNorm(s) / sqrtf(fabsf(Ps2FNorm(t))));
}
static inline float Ps2VRsqrtAt(uint32_t pc, float s, float t)
{
    if ((Ps2FloatBits(t) & 0x7F800000u) == 0u)
        Ps2NoteDivZero(pc);
    return Ps2VRsqrt(s, t);
}
// VU0 R register (stored splatted in ctx->vu0_r): 23-bit LFSR with exponent 0x3F800000.
static inline uint32_t Ps2VuRGet(__m128 r) { return static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_castps_si128(r))); }
static inline __m128 Ps2VuRSet(uint32_t v)
{
    return _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>((v & 0x007FFFFFu) | 0x3F800000u)));
}
static inline __m128 Ps2VuRNext(__m128 r)
{
    uint32_t v = Ps2VuRGet(r);
    const uint32_t x = (v >> 4) & 1u, y = (v >> 22) & 1u;
    v = (v << 1) ^ x ^ y;
    return Ps2VuRSet(v);
}

#define PS2_VADD(a, b) Ps2VAdd((__m128)(a), (__m128)(b))
#define PS2_VSUB(a, b) Ps2VSub((__m128)(a), (__m128)(b))
#define PS2_VMUL(a, b) Ps2VMul((__m128)(a), (__m128)(b))
#define PS2_VMADD(acc, a, b) Ps2VMadd((__m128)(acc), (__m128)(a), (__m128)(b))
#define PS2_VMSUB(acc, a, b) Ps2VMsub((__m128)(acc), (__m128)(a), (__m128)(b))
#define PS2_VMAX(a, b) Ps2VMax((__m128)(a), (__m128)(b))
#define PS2_VMINI(a, b) Ps2VMin((__m128)(a), (__m128)(b))
#define PS2_VDIV(a, b) Ps2VNorm(_mm_div_ps(Ps2VNorm((__m128)(a)), Ps2VNorm((__m128)(b))))
#define PS2_VMULQ(a, q) Ps2VMul((__m128)(a), _mm_set1_ps(q))
#define PS2_VBLEND(a, b, mask) PS2_BLENDV_PS((__m128)(a), (__m128)(b), (__m128)(mask))

// Memory access helpers - Hybrid Fast/Slow Path
// Fast path: Direct RDRAM access (masked).
// Slow path: Full runtime->Load/Store

static inline bool Ps2FastRangeIsContiguous(uint32_t offset, uint32_t bytes)
{
    return offset <= (PS2_RAM_SIZE - bytes);
}

static inline uint8_t Ps2FastRead8(const uint8_t *rdram, uint32_t addr)
{
    return rdram[addr & PS2_RAM_MASK];
}

static inline uint16_t Ps2FastRead16(const uint8_t *rdram, uint32_t addr)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint16_t)))
    {
        uint8_t wrapped[sizeof(uint16_t)];
        for (uint32_t i = 0; i < sizeof(uint16_t); ++i)
        {
            wrapped[i] = rdram[(offset + i) & PS2_RAM_MASK];
        }
        uint16_t value;
        std::memcpy(&value, wrapped, sizeof(value));
        return value;
    }

    uint16_t value;
    std::memcpy(&value, rdram + offset, sizeof(value));
    return value;
}

static inline uint32_t Ps2FastRead32(const uint8_t *rdram, uint32_t addr)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint32_t)))
    {
        uint8_t wrapped[sizeof(uint32_t)];
        for (uint32_t i = 0; i < sizeof(uint32_t); ++i)
        {
            wrapped[i] = rdram[(offset + i) & PS2_RAM_MASK];
        }
        uint32_t value;
        std::memcpy(&value, wrapped, sizeof(value));
        return value;
    }

    uint32_t value;
    std::memcpy(&value, rdram + offset, sizeof(value));
    return value;
}

static inline uint64_t Ps2FastRead64(const uint8_t *rdram, uint32_t addr)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint64_t)))
    {
        uint8_t wrapped[sizeof(uint64_t)];
        for (uint32_t i = 0; i < sizeof(uint64_t); ++i)
        {
            wrapped[i] = rdram[(offset + i) & PS2_RAM_MASK];
        }
        uint64_t value;
        std::memcpy(&value, wrapped, sizeof(value));
        return value;
    }

    uint64_t value;
    std::memcpy(&value, rdram + offset, sizeof(value));
    return value;
}

// All EE 128-bit accesses (LQ/SQ/LQC2/SQC2) silently ignore the low 4 address bits.
static inline __m128i Ps2FastRead128(const uint8_t *rdram, uint32_t addr)
{
    const uint32_t offset = addr & PS2_RAM_MASK & ~0xFu;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(__m128i)))
    {
        alignas(16) uint8_t wrapped[sizeof(__m128i)];
        for (uint32_t i = 0; i < sizeof(__m128i); ++i)
        {
            wrapped[i] = rdram[(offset + i) & PS2_RAM_MASK];
        }
        __m128i value;
        std::memcpy(&value, wrapped, sizeof(value));
        return value;
    }

    __m128i value;
    std::memcpy(&value, rdram + offset, sizeof(value));
    return value;
}

static inline void Ps2FastWrite8(uint8_t *rdram, uint32_t addr, uint8_t value)
{
    rdram[addr & PS2_RAM_MASK] = value;
}

static inline void Ps2FastWrite16(uint8_t *rdram, uint32_t addr, uint16_t value)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint16_t)))
    {
        uint8_t wrapped[sizeof(uint16_t)];
        std::memcpy(wrapped, &value, sizeof(value));
        for (uint32_t i = 0; i < sizeof(uint16_t); ++i)
        {
            rdram[(offset + i) & PS2_RAM_MASK] = wrapped[i];
        }
        return;
    }
    std::memcpy(rdram + offset, &value, sizeof(value));
}

static inline void Ps2FastWrite32(uint8_t *rdram, uint32_t addr, uint32_t value)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint32_t)))
    {
        uint8_t wrapped[sizeof(uint32_t)];
        std::memcpy(wrapped, &value, sizeof(value));
        for (uint32_t i = 0; i < sizeof(uint32_t); ++i)
        {
            rdram[(offset + i) & PS2_RAM_MASK] = wrapped[i];
        }
        return;
    }
    std::memcpy(rdram + offset, &value, sizeof(value));
}

static inline void Ps2FastWrite64(uint8_t *rdram, uint32_t addr, uint64_t value)
{
    const uint32_t offset = addr & PS2_RAM_MASK;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(uint64_t)))
    {
        uint8_t wrapped[sizeof(uint64_t)];
        std::memcpy(wrapped, &value, sizeof(value));
        for (uint32_t i = 0; i < sizeof(uint64_t); ++i)
        {
            rdram[(offset + i) & PS2_RAM_MASK] = wrapped[i];
        }
        return;
    }
    std::memcpy(rdram + offset, &value, sizeof(value));
}

static inline void Ps2FastWrite128(uint8_t *rdram, uint32_t addr, __m128i value)
{
    const uint32_t offset = addr & PS2_RAM_MASK & ~0xFu;
    if (!Ps2FastRangeIsContiguous(offset, sizeof(__m128i)))
    {
        alignas(16) uint8_t wrapped[sizeof(__m128i)];
        std::memcpy(wrapped, &value, sizeof(value));
        for (uint32_t i = 0; i < sizeof(__m128i); ++i)
        {
            rdram[(offset + i) & PS2_RAM_MASK] = wrapped[i];
        }
        return;
    }
    std::memcpy(rdram + offset, &value, sizeof(value));
}

#define FAST_READ8(addr) Ps2FastRead8(rdram, (uint32_t)(addr))
#define FAST_READ16(addr) Ps2FastRead16(rdram, (uint32_t)(addr))
#define FAST_READ32(addr) Ps2FastRead32(rdram, (uint32_t)(addr))
#define FAST_READ64(addr) Ps2FastRead64(rdram, (uint32_t)(addr))
#define FAST_READ128(addr) Ps2FastRead128(rdram, (uint32_t)(addr))

#define FAST_WRITE8(addr, val) Ps2FastWrite8(rdram, (uint32_t)(addr), (uint8_t)(val))
#define FAST_WRITE16(addr, val) Ps2FastWrite16(rdram, (uint32_t)(addr), (uint16_t)(val))
#define FAST_WRITE32(addr, val) Ps2FastWrite32(rdram, (uint32_t)(addr), (uint32_t)(val))
#define FAST_WRITE64(addr, val) Ps2FastWrite64(rdram, (uint32_t)(addr), (uint64_t)(val))
#define FAST_WRITE128(addr, val) Ps2FastWrite128(rdram, (uint32_t)(addr), (val))

#define READ8(addr) ([&]() -> uint8_t {                       \
    uint32_t _addr = (uint32_t)(addr);                        \
    return PS2Runtime::isSpecialAddress(_addr)                \
        ? runtime->Load8(rdram, ctx, _addr)                   \
        : FAST_READ8(_addr); }())

#define READ16(addr) ([&]() -> uint16_t {                     \
    uint32_t _addr = (uint32_t)(addr);                        \
    return PS2Runtime::isSpecialAddress(_addr)                \
        ? runtime->Load16(rdram, ctx, _addr)                  \
        : FAST_READ16(_addr); }())

#define READ32(addr) ([&]() -> uint32_t {                     \
    uint32_t _addr = (uint32_t)(addr);                        \
    return PS2Runtime::isSpecialAddress(_addr)                \
        ? runtime->Load32(rdram, ctx, _addr)                  \
        : FAST_READ32(_addr); }())

#define READ64(addr) ([&]() -> uint64_t {                     \
    uint32_t _addr = (uint32_t)(addr);                        \
    return PS2Runtime::isSpecialAddress(_addr)                \
        ? runtime->Load64(rdram, ctx, _addr)                  \
        : FAST_READ64(_addr); }())

#define READ128(addr) ([&]() -> __m128i {                     \
    uint32_t _addr = (uint32_t)(addr) & ~0xFu; /* LQ aligns */ \
    return PS2Runtime::isSpecialAddress(_addr)                \
        ? runtime->Load128(rdram, ctx, _addr)                 \
        : FAST_READ128(_addr); }())

#define WRITE8(addr, val)                                                            \
    do                                                                               \
    {                                                                                \
        uint32_t _addr = (addr);                                                     \
        if (PS2Runtime::isSpecialAddress(_addr))                                     \
            runtime->Store8(rdram, ctx, _addr, (val));                               \
        else                                                                         \
        {                                                                            \
            ps2TraceGuestWrite(rdram, _addr, 1u, (uint8_t)(val), 0u, "WRITE8", ctx); \
            FAST_WRITE8(_addr, (val));                                               \
        }                                                                            \
    } while (0)

#define WRITE16(addr, val)                                                             \
    do                                                                                 \
    {                                                                                  \
        uint32_t _addr = (addr);                                                       \
        if (PS2Runtime::isSpecialAddress(_addr))                                       \
            runtime->Store16(rdram, ctx, _addr, (val));                                \
        else                                                                           \
        {                                                                              \
            ps2TraceGuestWrite(rdram, _addr, 2u, (uint16_t)(val), 0u, "WRITE16", ctx); \
            FAST_WRITE16(_addr, (val));                                                \
        }                                                                              \
    } while (0)

#define WRITE32(addr, val)                                                             \
    do                                                                                 \
    {                                                                                  \
        uint32_t _addr = (addr);                                                       \
        if (PS2Runtime::isSpecialAddress(_addr))                                       \
            runtime->Store32(rdram, ctx, _addr, (val));                                \
        else                                                                           \
        {                                                                              \
            ps2TraceGuestWrite(rdram, _addr, 4u, (uint32_t)(val), 0u, "WRITE32", ctx); \
            FAST_WRITE32(_addr, (val));                                                \
        }                                                                              \
    } while (0)

#define WRITE64(addr, val)                                                             \
    do                                                                                 \
    {                                                                                  \
        uint32_t _addr = (addr);                                                       \
        if (PS2Runtime::isSpecialAddress(_addr))                                       \
            runtime->Store64(rdram, ctx, _addr, (val));                                \
        else                                                                           \
        {                                                                              \
            ps2TraceGuestWrite(rdram, _addr, 8u, (uint64_t)(val), 0u, "WRITE64", ctx); \
            FAST_WRITE64(_addr, (val));                                                \
        }                                                                              \
    } while (0)

#define WRITE128(addr, val)                                                          \
    do                                                                               \
    {                                                                                \
        uint32_t _addr = (uint32_t)(addr) & ~0xFu; /* SQ aligns */                   \
        __m128i _value = (val);                                                      \
        if (PS2Runtime::isSpecialAddress(_addr))                                     \
            runtime->Store128(rdram, ctx, _addr, _value);                            \
        else                                                                         \
        {                                                                            \
            const uint64_t _lo = static_cast<uint64_t>(PS2_EXTRACT_EPI64_0(_value)); \
            const uint64_t _hi = static_cast<uint64_t>(PS2_EXTRACT_EPI64_1(_value)); \
            ps2TraceGuestWrite(rdram, _addr, 16u, _lo, _hi, "WRITE128", ctx);        \
            FAST_WRITE128(_addr, _value);                                            \
        }                                                                            \
    } while (0)

// Packed Compare Greater Than (PCGT)
#define PS2_PCGTW(a, b) _mm_cmpgt_epi32((__m128i)(a), (__m128i)(b))
#define PS2_PCGTH(a, b) _mm_cmpgt_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PCGTB(a, b) _mm_cmpgt_epi8((__m128i)(a), (__m128i)(b))

// Packed Add with Signed Saturation Word (PADDSW)
inline __m128i ps2_paddsw(__m128i a, __m128i b)
{

    __m128i sum = _mm_add_epi32(a, b);
    // Check for over/underflow. Clamp to either INT32_MIN/INT32_MAX.
    __m128i overflow = _mm_and_si128(_mm_xor_si128(a, sum),
                                     _mm_xor_si128(b, sum));
    // Extract input sign.
    overflow = _mm_srai_epi32(overflow, 31);
    __m128i input_sign = _mm_srai_epi32(a, 31);
    // Select saturation value based on overflow sign.
    #if defined(__SSE4_1__)
    __m128i sat = _mm_blendv_epi8(
        _mm_set1_epi32(INT32_MAX),
        _mm_set1_epi32(INT32_MIN),
        input_sign);
    return _mm_blendv_epi8(sum, sat, overflow);
    #else
    __m128i sat = _mm_or_si128(_mm_and_si128(input_sign, _mm_set1_epi32(INT32_MIN)),
                               _mm_andnot_si128(input_sign, _mm_set1_epi32(INT32_MAX)));
    return _mm_or_si128(_mm_and_si128(overflow, sat),
                        _mm_andnot_si128(overflow, sum));
    #endif
}
#define PS2_PADDSW(a, b) ps2_paddsw((__m128i)(a), (__m128i)(b))

// Packed Subtract with Signed Saturation Word (PSUBSW)
inline __m128i ps2_psubsw(__m128i a, __m128i b)
{
    __m128i diff = _mm_sub_epi32(a, b);
    // Check for over/underflow. Clamp to either INT32_MIN/INT32_MAX.
    __m128i overflow = _mm_and_si128(_mm_xor_si128(a, b),
                                     _mm_xor_si128(a, diff));
    // Extract input sign.
    overflow = _mm_srai_epi32(overflow, 31);
    __m128i input_sign = _mm_srai_epi32(a, 31);
    // Select saturation value based on overflow sign.
    #if defined(__SSE4_1__)
    __m128i sat = _mm_blendv_epi8(
        _mm_set1_epi32(INT32_MAX),
        _mm_set1_epi32(INT32_MIN),
        input_sign);
    return _mm_blendv_epi8(diff, sat, overflow);
    #else
    __m128i sat = _mm_or_si128(_mm_and_si128(input_sign, _mm_set1_epi32(INT32_MIN)),
                               _mm_andnot_si128(input_sign, _mm_set1_epi32(INT32_MAX)));
    return _mm_or_si128(_mm_and_si128(overflow, sat),
                        _mm_andnot_si128(overflow, diff));
    #endif

}
#define PS2_PSUBSW(a, b) ps2_psubsw((__m128i)(a), (__m128i)(b))

// Packed Compare Equal (PCEQ)
#define PS2_PCEQW(a, b) _mm_cmpeq_epi32((__m128i)(a), (__m128i)(b))
#define PS2_PCEQH(a, b) _mm_cmpeq_epi16((__m128i)(a), (__m128i)(b))
#define PS2_PCEQB(a, b) _mm_cmpeq_epi8((__m128i)(a), (__m128i)(b))

// Packed Absolute (PABS). The EE saturates: |0x80000000| = 0x7FFFFFFF, |0x8000| = 0x7FFF.
#define PS2_PABSW(a) _mm_min_epu32(_mm_abs_epi32((__m128i)(a)), _mm_set1_epi32(0x7FFFFFFF))
#define PS2_PABSH(a) _mm_min_epu16(_mm_abs_epi16((__m128i)(a)), _mm_set1_epi16(0x7FFF))

// Packed Pack (PPAC) - Packs larger elements into smaller ones
inline __m128i ps2_paddu32(__m128i a, __m128i b)
{
    __m128i sum = _mm_add_epi32(a, b);
    __m128i overflow = _mm_cmpgt_epi32(_mm_xor_si128(a, _mm_set1_epi32(INT32_MIN)),
                                       _mm_xor_si128(sum, _mm_set1_epi32(INT32_MIN)));
    return _mm_or_si128(sum, overflow); // overflow lanes become all-1s
}
inline __m128i ps2_psubu32(__m128i a, __m128i b)
{
    __m128i diff = _mm_sub_epi32(a, b);
    // Underflow if a < b (unsigned). Clamp to 0.
    __m128i underflow = _mm_cmpgt_epi32(_mm_xor_si128(b, _mm_set1_epi32(INT32_MIN)),
                                        _mm_xor_si128(a, _mm_set1_epi32(INT32_MIN)));
    return _mm_andnot_si128(underflow, diff); // underflow lanes become 0
}

inline __m128i ps2_ppacw(__m128i rs, __m128i rt)
{
    // rs = [rs3 rs2 rs1 rs0], rt = [rt3 rt2 rt1 rt0]
    return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(rt), _mm_castsi128_ps(rs), _MM_SHUFFLE(2, 0, 2, 0)));
}
#define PS2_PPACW(a, b) ps2_ppacw((__m128i)(a), (__m128i)(b))

inline __m128i ps2_ppach(__m128i rs, __m128i rt)
{
    const __m128i mask = _mm_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13,  // from rt: halfwords 0,2,4,6
        0, 1, 4, 5, 8, 9, 12, 13); // from rs: halfwords 0,2,4,6
    __m128i lo = _mm_shuffle_epi8(rt, mask);
    __m128i hi = _mm_shuffle_epi8(rs, mask);
    return _mm_unpacklo_epi64(lo, hi);
}
#define PS2_PPACH(a, b) ps2_ppach((__m128i)(a), (__m128i)(b))

inline __m128i ps2_ppacb(__m128i rs, __m128i rt)
{
    const __m128i mask = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14,  // from rt: bytes 0,2,4,6,8,10,12,14
        0, 2, 4, 6, 8, 10, 12, 14); // from rs
    __m128i lo = _mm_shuffle_epi8(rt, mask);
    __m128i hi = _mm_shuffle_epi8(rs, mask);
    return _mm_unpacklo_epi64(lo, hi);
}
#define PS2_PPACB(a, b) ps2_ppacb((__m128i)(a), (__m128i)(b))

// Packed Interleave (a = rs, b = rt)
// PINTH:  rd.h = {rt.h0, rs.h4, rt.h1, rs.h5, rt.h2, rs.h6, rt.h3, rs.h7}
// PINTEH: rd.h = {rt.h0, rs.h0, rt.h2, rs.h2, rt.h4, rs.h4, rt.h6, rs.h6}
#define PS2_PINTH(a, b) _mm_unpacklo_epi16((__m128i)(b), _mm_unpackhi_epi64((__m128i)(a), (__m128i)(a)))
#define PS2_PINTEH(a, b) _mm_or_si128(_mm_and_si128((__m128i)(b), _mm_set1_epi32(0xFFFF)), _mm_slli_epi32((__m128i)(a), 16))

// Packed Variable Shifts (a = rs = shift amounts, b = rt = values). Only words 0 and 2 are shifted;
// each 32-bit result is sign-extended into its 64-bit half.
#define PS2_PSLLVW(a, b) Ps2MmiShiftVW((__m128i)(b), (__m128i)(a), 0)
#define PS2_PSRLVW(a, b) Ps2MmiShiftVW((__m128i)(b), (__m128i)(a), 1)
#define PS2_PSRAVW(a, b) Ps2MmiShiftVW((__m128i)(b), (__m128i)(a), 2)

static inline __m128i Ps2MmiShiftVW(__m128i rt, __m128i rs, int kind)
{
    alignas(16) uint32_t t[4], s[4];
    alignas(16) int64_t r[2];
    _mm_store_si128((__m128i *)t, rt);
    _mm_store_si128((__m128i *)s, rs);
    for (int i = 0; i < 2; i++)
    {
        const uint32_t v = t[i * 2], n = s[i * 2] & 0x1Fu;
        const uint32_t w = kind == 0 ? (v << n) : kind == 1 ? (v >> n) : (uint32_t)((int32_t)v >> n);
        r[i] = (int64_t)(int32_t)w;
    }
    return _mm_load_si128((const __m128i *)r);
}

// ---------------------------------------------------------------------------------------------
// R5900 MMI HI/LO helpers. Semantics follow PCSX2 (pcsx2/MMI.cpp; PMADDW/PMSUBW follow its x86
// recompiler, which does exact 64-bit HI:LO +/- product).
// HI and LO are 128-bit registers on the EE: ctx->lo/ctx->hi hold bits 0-63 (the MULT/DIV pipe 0
// result), ctx->lo1/ctx->hi1 hold bits 64-127 (pipe 1: MULT1/DIV1). Word n of LO is LO.UL[n] below.
// ---------------------------------------------------------------------------------------------
union Ps2MmiQ
{
    uint8_t ub[16];
    uint16_t uh[8];
    int16_t sh[8];
    uint32_t uw[4];
    int32_t sw[4];
    uint64_t ud[2];
    int64_t sd[2];
};
static inline Ps2MmiQ Ps2MmiFromVec(__m128i v)
{
    Ps2MmiQ q;
    std::memcpy(&q, &v, sizeof(q));
    return q;
}
static inline __m128i Ps2MmiToVec(const Ps2MmiQ &q)
{
    __m128i v;
    std::memcpy(&v, &q, sizeof(v));
    return v;
}
static inline Ps2MmiQ Ps2MmiGetLO(const R5900Context *ctx) { Ps2MmiQ q; q.ud[0] = ctx->lo; q.ud[1] = ctx->lo1; return q; }
static inline Ps2MmiQ Ps2MmiGetHI(const R5900Context *ctx) { Ps2MmiQ q; q.ud[0] = ctx->hi; q.ud[1] = ctx->hi1; return q; }
static inline void Ps2MmiSetLO(R5900Context *ctx, const Ps2MmiQ &q) { ctx->lo = q.ud[0]; ctx->lo1 = q.ud[1]; }
static inline void Ps2MmiSetHI(R5900Context *ctx, const Ps2MmiQ &q) { ctx->hi = q.ud[0]; ctx->hi1 = q.ud[1]; }

// PMFHI / PMFLO / PMTHI / PMTLO: full 128-bit moves.
#define PS2_PMFHI(ctx) _mm_set_epi64x((long long)(ctx)->hi1, (long long)(ctx)->hi)
#define PS2_PMFLO(ctx) _mm_set_epi64x((long long)(ctx)->lo1, (long long)(ctx)->lo)
static inline void Ps2MmiPMTHI(R5900Context *ctx, __m128i v) { Ps2MmiSetHI(ctx, Ps2MmiFromVec(v)); }
static inline void Ps2MmiPMTLO(R5900Context *ctx, __m128i v) { Ps2MmiSetLO(ctx, Ps2MmiFromVec(v)); }

// PMFHL.fmt (fmt = sa field: 0 LW, 1 UW, 2 SLW, 3 LH, 4 SH)
static inline __m128i Ps2MmiPMFHL(const R5900Context *ctx, int fmt)
{
    const Ps2MmiQ lo = Ps2MmiGetLO(ctx), hi = Ps2MmiGetHI(ctx);
    Ps2MmiQ d{};
    auto clampH = [](int32_t v) -> uint16_t { return v > 0x7FFF ? 0x7FFF : v < -0x8000 ? 0x8000 : (uint16_t)v; };
    switch (fmt)
    {
    case 0: d.uw[0] = lo.uw[0]; d.uw[1] = hi.uw[0]; d.uw[2] = lo.uw[2]; d.uw[3] = hi.uw[2]; break;
    case 1: d.uw[0] = lo.uw[1]; d.uw[1] = hi.uw[1]; d.uw[2] = lo.uw[3]; d.uw[3] = hi.uw[3]; break;
    case 2:
        for (int i = 0; i < 2; i++)
        {
            const int64_t v = (int64_t)(((uint64_t)hi.uw[2 * i] << 32) | lo.uw[2 * i]);
            d.sd[i] = v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : v;
        }
        break;
    case 3:
        d.uh[0] = lo.uh[0]; d.uh[1] = lo.uh[2]; d.uh[2] = hi.uh[0]; d.uh[3] = hi.uh[2];
        d.uh[4] = lo.uh[4]; d.uh[5] = lo.uh[6]; d.uh[6] = hi.uh[4]; d.uh[7] = hi.uh[6];
        break;
    default:
        d.uh[0] = clampH(lo.sw[0]); d.uh[1] = clampH(lo.sw[1]); d.uh[2] = clampH(hi.sw[0]); d.uh[3] = clampH(hi.sw[1]);
        d.uh[4] = clampH(lo.sw[2]); d.uh[5] = clampH(lo.sw[3]); d.uh[6] = clampH(hi.sw[2]); d.uh[7] = clampH(hi.sw[3]);
        break;
    }
    return Ps2MmiToVec(d);
}
// PMTHL.LW: LO.w0 = rs.w0, HI.w0 = rs.w1, LO.w2 = rs.w2, HI.w2 = rs.w3 (odd words of HI/LO unchanged).
static inline void Ps2MmiPMTHL(R5900Context *ctx, __m128i v)
{
    const Ps2MmiQ s = Ps2MmiFromVec(v);
    Ps2MmiQ lo = Ps2MmiGetLO(ctx), hi = Ps2MmiGetHI(ctx);
    lo.uw[0] = s.uw[0]; hi.uw[0] = s.uw[1]; lo.uw[2] = s.uw[2]; hi.uw[2] = s.uw[3];
    Ps2MmiSetLO(ctx, lo);
    Ps2MmiSetHI(ctx, hi);
}
// Legacy names (older generated code passed only the low 64 bits; ctx is in scope there).
#define PS2_PMFHL_LW(hi, lo) Ps2MmiPMFHL(ctx, 0)
#define PS2_PMFHL_UW(hi, lo) Ps2MmiPMFHL(ctx, 1)
#define PS2_PMFHL_SLW(hi, lo) Ps2MmiPMFHL(ctx, 2)
#define PS2_PMFHL_LH(hi, lo) Ps2MmiPMFHL(ctx, 3)
#define PS2_PMFHL_SH(hi, lo) Ps2MmiPMFHL(ctx, 4)

// PMULTW (0), PMADDW (1), PMSUBW (2), PMULTUW (3), PMADDUW (4): word lanes 0 and 2.
// LO/HI doubleword i = sign-extended low/high 32 bits of the 64-bit result; rd doubleword i = result.
static inline __m128i Ps2MmiMulW(R5900Context *ctx, __m128i rsV, __m128i rtV, int mode)
{
    const Ps2MmiQ a = Ps2MmiFromVec(rsV), b = Ps2MmiFromVec(rtV);
    Ps2MmiQ lo = Ps2MmiGetLO(ctx), hi = Ps2MmiGetHI(ctx), d;
    for (int i = 0; i < 2; i++)
    {
        const int w = i * 2;
        const uint64_t acc = ((uint64_t)hi.uw[w] << 32) | lo.uw[w];
        uint64_t r;
        if (mode >= 3)
        {
            const uint64_t p = (uint64_t)a.uw[w] * b.uw[w];
            r = mode == 3 ? p : acc + p;
        }
        else
        {
            const uint64_t p = (uint64_t)((int64_t)a.sw[w] * (int64_t)b.sw[w]);
            r = mode == 0 ? p : mode == 1 ? acc + p : acc - p;
        }
        lo.sd[i] = (int32_t)(uint32_t)r;
        hi.sd[i] = (int32_t)(uint32_t)(r >> 32);
        d.ud[i] = r;
    }
    Ps2MmiSetLO(ctx, lo);
    Ps2MmiSetHI(ctx, hi);
    return Ps2MmiToVec(d);
}

// PDIVW / PDIVUW: word lanes 0 and 2, quotient -> LO doubleword i, remainder -> HI (sign-extended).
// Divide by zero: LO = (rs < 0 ? 1 : -1) (PDIVUW: -1), HI = rs. 0x80000000 / -1 = 0x80000000 rem 0.
static inline void Ps2MmiDivW(R5900Context *ctx, __m128i rsV, __m128i rtV, bool isUnsigned)
{
    const Ps2MmiQ a = Ps2MmiFromVec(rsV), b = Ps2MmiFromVec(rtV);
    Ps2MmiQ lo = Ps2MmiGetLO(ctx), hi = Ps2MmiGetHI(ctx);
    for (int i = 0; i < 2; i++)
    {
        const int w = i * 2;
        if (isUnsigned)
        {
            if (b.uw[w] != 0) { lo.sd[i] = (int32_t)(a.uw[w] / b.uw[w]); hi.sd[i] = (int32_t)(a.uw[w] % b.uw[w]); }
            else { lo.sd[i] = -1; hi.sd[i] = a.sw[w]; }
        }
        else if (a.uw[w] == 0x80000000u && b.uw[w] == 0xFFFFFFFFu) { lo.sd[i] = INT32_MIN; hi.sd[i] = 0; }
        else if (b.sw[w] != 0) { lo.sd[i] = a.sw[w] / b.sw[w]; hi.sd[i] = a.sw[w] % b.sw[w]; }
        else { lo.sd[i] = a.sw[w] < 0 ? 1 : -1; hi.sd[i] = a.sw[w]; }
    }
    Ps2MmiSetLO(ctx, lo);
    Ps2MmiSetHI(ctx, hi);
}

// PDIVBW: each word of rs / rt.h0 (signed); quotient -> LO.w[n], remainder -> HI.w[n]. No rd write.
static inline void Ps2MmiPDIVBW(R5900Context *ctx, __m128i rsV, __m128i rtV)
{
    const Ps2MmiQ a = Ps2MmiFromVec(rsV), b = Ps2MmiFromVec(rtV);
    Ps2MmiQ lo, hi;
    const int32_t div = b.sh[0];
    for (int n = 0; n < 4; n++)
    {
        if (a.uw[n] == 0x80000000u && div == -1) { lo.sw[n] = INT32_MIN; hi.sw[n] = 0; }
        else if (div != 0) { lo.sw[n] = a.sw[n] / div; hi.sw[n] = a.sw[n] % div; }
        else { lo.sw[n] = a.sw[n] < 0 ? 1 : -1; hi.sw[n] = a.sw[n]; }
    }
    Ps2MmiSetLO(ctx, lo);
    Ps2MmiSetHI(ctx, hi);
}

// PMULTH (0), PMADDH (1), PMSUBH (2): halfword products p0..p7 go (32-bit, wrapping) to
// {LO.w0, LO.w1, HI.w0, HI.w1, LO.w2, LO.w3, HI.w2, HI.w3}; rd = {LO.w0, HI.w0, LO.w2, HI.w2}.
static inline __m128i Ps2MmiMulH(R5900Context *ctx, __m128i rsV, __m128i rtV, int mode)
{
    const Ps2MmiQ a = Ps2MmiFromVec(rsV), b = Ps2MmiFromVec(rtV);
    Ps2MmiQ lo = Ps2MmiGetLO(ctx), hi = Ps2MmiGetHI(ctx), d;
    uint32_t *dst[8] = {&lo.uw[0], &lo.uw[1], &hi.uw[0], &hi.uw[1], &lo.uw[2], &lo.uw[3], &hi.uw[2], &hi.uw[3]};
    for (int n = 0; n < 8; n++)
    {
        const uint32_t p = (uint32_t)((int32_t)a.sh[n] * (int32_t)b.sh[n]);
        *dst[n] = mode == 0 ? p : mode == 1 ? *dst[n] + p : *dst[n] - p;
    }
    d.uw[0] = lo.uw[0]; d.uw[1] = hi.uw[0]; d.uw[2] = lo.uw[2]; d.uw[3] = hi.uw[2];
    Ps2MmiSetLO(ctx, lo);
    Ps2MmiSetHI(ctx, hi);
    return Ps2MmiToVec(d);
}

// PHMADH / PHMSBH: per word pair k (halfwords 2k, 2k+1), f = rs.h[2k+1]*rt.h[2k+1], e = rs.h[2k]*rt.h[2k].
// Sum (f + e, or f - e for PHMSBH) -> {LO.w0, HI.w0, LO.w2, HI.w2}[k]; the odd words get f (PHMADH) or
// ~f (PHMSBH) - undocumented, matches PCSX2. rd = the four sums.
static inline __m128i Ps2MmiHMulH(R5900Context *ctx, __m128i rsV, __m128i rtV, bool subtract)
{
    const Ps2MmiQ a = Ps2MmiFromVec(rsV), b = Ps2MmiFromVec(rtV);
    Ps2MmiQ lo = Ps2MmiGetLO(ctx), hi = Ps2MmiGetHI(ctx), d;
    uint32_t *dst[4] = {&lo.uw[0], &hi.uw[0], &lo.uw[2], &hi.uw[2]};
    for (int k = 0; k < 4; k++)
    {
        const uint32_t f = (uint32_t)((int32_t)a.sh[2 * k + 1] * (int32_t)b.sh[2 * k + 1]);
        const uint32_t e = (uint32_t)((int32_t)a.sh[2 * k] * (int32_t)b.sh[2 * k]);
        dst[k][0] = subtract ? f - e : f + e;
        dst[k][1] = subtract ? ~f : f;
        d.uw[k] = dst[k][0];
    }
    Ps2MmiSetLO(ctx, lo);
    Ps2MmiSetHI(ctx, hi);
    return Ps2MmiToVec(d);
}

// FPU (COP1) operations (PCSX2 FPU.cpp semantics: operands and results normalised, see Ps2FNorm)
#define FPU_SET_ACC(ctx, res) (ctx->f_acc = res)
#define FPU_ADD_S(a, b) Ps2FNorm(Ps2FNorm((float)(a)) + Ps2FNorm((float)(b)))
#define FPU_SUB_S(a, b) Ps2FNorm(Ps2FNorm((float)(a)) - Ps2FNorm((float)(b)))
#define FPU_MUL_S(a, b) Ps2FNorm(Ps2FNorm((float)(a)) * Ps2FNorm((float)(b)))
#define FPU_DIV_S(a, b) Ps2FDiv((float)(a), (float)(b))
// sqrt.s fd, ft: +-0/denormal -> signed zero, otherwise sqrt(|ft|)
static inline float Ps2FSqrt(float t)
{
    const uint32_t u = Ps2FloatBits(t);
    if ((u & 0x7F800000u) == 0u)
        return Ps2BitsFloat(u & 0x80000000u);
    return sqrtf(fabsf(Ps2FNorm(t)));
}
// rsqrt.s fd, fs, ft: fs/sqrt(|ft|); ft==0 -> +-FLT_MAX with the sign of ft
static inline float Ps2FRsqrt(float s, float t)
{
    const uint32_t ut = Ps2FloatBits(t);
    if ((ut & 0x7F800000u) == 0u)
        return Ps2BitsFloat((ut & 0x80000000u) | 0x7F7FFFFFu);
    return Ps2FNorm(Ps2FNorm(s) / sqrtf(fabsf(Ps2FNorm(t))));
}
// cvt.w.s truncates; |x| >= 2^31 (incl. Inf/NaN patterns) saturates by sign
static inline int32_t Ps2FCvtWS(float v)
{
    const uint32_t u = Ps2FloatBits(v);
    if ((u & 0x7F800000u) <= 0x4E800000u)
        return static_cast<int32_t>(v);
    return (u & 0x80000000u) ? static_cast<int32_t>(0x80000000u) : 0x7FFFFFFF;
}
// max.s/min.s compare the raw bit patterns as sign-magnitude integers
static inline float Ps2FMax(float a, float b)
{
    const int32_t ia = static_cast<int32_t>(Ps2FloatBits(a)), ib = static_cast<int32_t>(Ps2FloatBits(b));
    const int32_t r = (ia < 0 && ib < 0) ? (ia < ib ? ia : ib) : (ia > ib ? ia : ib);
    return Ps2BitsFloat(static_cast<uint32_t>(r));
}
static inline float Ps2FMin(float a, float b)
{
    const int32_t ia = static_cast<int32_t>(Ps2FloatBits(a)), ib = static_cast<int32_t>(Ps2FloatBits(b));
    const int32_t r = (ia < 0 && ib < 0) ? (ia > ib ? ia : ib) : (ia < ib ? ia : ib);
    return Ps2BitsFloat(static_cast<uint32_t>(r));
}
#define FPU_SQRT_S(a) Ps2FSqrt((float)(a))
#define FPU_RSQRT_S(a, b) Ps2FRsqrt((float)(a), (float)(b))
#define FPU_MAX_S(a, b) Ps2FMax((float)(a), (float)(b))
#define FPU_MIN_S(a, b) Ps2FMin((float)(a), (float)(b))
#define FPU_ABS_S(a) Ps2BitsFloat(Ps2FloatBits((float)(a)) & 0x7FFFFFFFu)
#define FPU_MOV_S(a) ((float)(a))
#define FPU_NEG_S(a) Ps2BitsFloat(Ps2FloatBits((float)(a)) ^ 0x80000000u)
#define FPU_ROUND_L_S(a) ((int64_t)roundf((float)(a)))
#define FPU_TRUNC_L_S(a) ((int64_t)(float)(a))
#define FPU_CEIL_L_S(a) ((int64_t)ceilf((float)(a)))
#define FPU_FLOOR_L_S(a) ((int64_t)floorf((float)(a)))
#define FPU_ROUND_W_S(a) ((int32_t)nearbyintf((float)(a)))
#define FPU_TRUNC_W_S(a) ((int32_t)(float)(a))
#define FPU_CEIL_W_S(a) ((int32_t)ceilf((float)(a)))
#define FPU_FLOOR_W_S(a) ((int32_t)floorf((float)(a)))
#define FPU_CVT_S_W(a) ((float)(int32_t)(a))
#define FPU_CVT_S_L(a) ((float)(int64_t)(a))
#define FPU_CVT_W_S(a) Ps2FCvtWS((float)(a))
#define FPU_CVT_L_S(a) ((int64_t)(float)(a))
// c.cond.s: the R5900 has only C.F/C.EQ/C.LT/C.LE; operands are normalised (no NaN, denormal == 0),
// so the IEEE "unordered" variants reduce to the ordered compare.
#define FPU_C_F_S(a, b) (0)
#define FPU_C_UN_S(a, b) (0)
#define FPU_C_EQ_S(a, b) (Ps2FNorm((float)(a)) == Ps2FNorm((float)(b)))
#define FPU_C_UEQ_S(a, b) FPU_C_EQ_S(a, b)
#define FPU_C_OLT_S(a, b) (Ps2FNorm((float)(a)) < Ps2FNorm((float)(b)))
#define FPU_C_ULT_S(a, b) FPU_C_OLT_S(a, b)
#define FPU_C_OLE_S(a, b) (Ps2FNorm((float)(a)) <= Ps2FNorm((float)(b)))
#define FPU_C_ULE_S(a, b) FPU_C_OLE_S(a, b)
#define FPU_C_SF_S(a, b) (0)
#define FPU_C_NGLE_S(a, b) (0)
#define FPU_C_SEQ_S(a, b) FPU_C_EQ_S(a, b)
#define FPU_C_NGL_S(a, b) FPU_C_EQ_S(a, b)
#define FPU_C_LT_S(a, b) FPU_C_OLT_S(a, b)
#define FPU_C_NGE_S(a, b) FPU_C_OLT_S(a, b)
#define FPU_C_LE_S(a, b) FPU_C_OLE_S(a, b)
#define FPU_C_NGT_S(a, b) FPU_C_OLE_S(a, b)

// QFSRV: Quadword Funnel Shift Right Variable
// Concatenates rs || rt (256 bits) and right-shifts by SA bits, taking lower 128 bits.
inline __m128i ps2_qfsrv(__m128i rs, __m128i rt, uint32_t sa)
{
    if (sa == 0)
        return rt;
    if (sa >= 128)
    {
        if (sa >= 256)
            return _mm_setzero_si128();
        uint32_t shift = sa - 128;
        if (shift == 0)
            return rs;
        // Shift rs right by (sa-128) bits
        uint32_t byteShift = shift / 8;
        uint32_t bitShift = shift % 8;
        // Byte shift rs right
        alignas(16) uint8_t buf[16] = {};
        alignas(16) uint8_t src[16];
        _mm_store_si128((__m128i *)src, rs);
        for (uint32_t i = 0; i + byteShift < 16; i++)
            buf[i] = src[i + byteShift];
        __m128i result = _mm_load_si128((__m128i *)buf);
        if (bitShift > 0)
            result = _mm_or_si128(_mm_srli_epi64(result, bitShift),
                                  _mm_slli_epi64(_mm_bsrli_si128(result, 8), 64 - bitShift));
        return result;
    }
    // sa is 1..127: result = (rs || rt) >> sa, lower 128 bits
    uint32_t byteShift = sa / 8;
    uint32_t bitShift = sa % 8;
    alignas(16) uint8_t combined[32];
    _mm_store_si128((__m128i *)(combined), rt);      // low 128 bits
    _mm_store_si128((__m128i *)(combined + 16), rs); // high 128 bits
    // Shift right by byteShift bytes
    alignas(16) uint8_t shifted[16];
    for (uint32_t i = 0; i < 16; i++)
        shifted[i] = (i + byteShift < 32) ? combined[i + byteShift] : 0;
    __m128i result = _mm_load_si128((__m128i *)shifted);
    if (bitShift > 0)
    {
        uint8_t extra = (byteShift + 16 < 32) ? combined[byteShift + 16] : 0;
        __m128i hi_byte = _mm_insert_epi8(_mm_setzero_si128(), extra, 15);
        alignas(16) uint8_t src32[32];
        for (uint32_t i = 0; i < 32; i++)
            src32[i] = combined[i];
        uint64_t lo0, lo1, hi0, hi1;
        std::memcpy(&lo0, src32, 8);
        std::memcpy(&lo1, src32 + 8, 8);
        std::memcpy(&hi0, src32 + 16, 8);
        std::memcpy(&hi1, src32 + 24, 8);
        // 256-bit right shift by sa bits
        uint64_t r0, r1;
        if (sa < 64)
        {
            r0 = (lo0 >> sa) | (lo1 << (64 - sa));
            r1 = (lo1 >> sa) | (hi0 << (64 - sa));
        }
        else if (sa < 128)
        {
            uint32_t s = sa - 64;
            if (s == 0)
            {
                r0 = lo1;
                r1 = hi0;
            }
            else
            {
                r0 = (lo1 >> s) | (hi0 << (64 - s));
                r1 = (hi0 >> s) | (hi1 << (64 - s));
            }
        }
        else
        {
            r0 = 0;
            r1 = 0; // handled above
        }
        result = _mm_set_epi64x((long long)r1, (long long)r0);
    }
    return result;
}
#define PS2_QFSRV(rs, rt, sa) ps2_qfsrv((__m128i)(rs), (__m128i)(rt), (uint32_t)(sa))
#define PS2_PCPYLD(rs, rt) _mm_unpacklo_epi64(rt, rs)
// Halfword/word permutes of rt (per 64-bit half for the halfword forms):
// PEXEH {2,1,0,3}  PREVH {3,2,1,0}  PEXCH {0,2,1,3}  PEXEW {2,1,0,3}  PEXCW {0,2,1,3}  PROT3W {1,2,0,3}
#define PS2_PEXEH(rt) _mm_shufflelo_epi16(_mm_shufflehi_epi16((rt), _MM_SHUFFLE(3, 0, 1, 2)), _MM_SHUFFLE(3, 0, 1, 2))
#define PS2_PREVH(rt) _mm_shufflelo_epi16(_mm_shufflehi_epi16((rt), _MM_SHUFFLE(0, 1, 2, 3)), _MM_SHUFFLE(0, 1, 2, 3))
#define PS2_PEXCH(rt) _mm_shufflelo_epi16(_mm_shufflehi_epi16((rt), _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0))
#define PS2_PEXEW(rt) _mm_shuffle_epi32((rt), _MM_SHUFFLE(3, 0, 1, 2))
#define PS2_PEXCW(rt) _mm_shuffle_epi32((rt), _MM_SHUFFLE(3, 1, 2, 0))
#define PS2_PROT3W(rt) _mm_shuffle_epi32((rt), _MM_SHUFFLE(3, 0, 2, 1))

// Additional VU0 operations
#define PS2_VSQRT(x) Ps2VSqrt(x)
#define PS2_VRSQRT(x) Ps2VRsqrt(1.0f, (x))

#define GPR_U32(ctx_ptr, reg_idx) ((reg_idx == 0) ? 0U : static_cast<uint32_t>(PS2_EXTRACT_EPI32_0(ctx_ptr->r[reg_idx])))
#define GPR_S32(ctx_ptr, reg_idx) ((reg_idx == 0) ? 0 : PS2_EXTRACT_EPI32_0(ctx_ptr->r[reg_idx]))
#define GPR_U64(ctx_ptr, reg_idx) ((reg_idx == 0) ? 0ULL : static_cast<uint64_t>(PS2_EXTRACT_EPI64_0(ctx_ptr->r[reg_idx])))
#define GPR_S64(ctx_ptr, reg_idx) ((reg_idx == 0) ? 0LL : PS2_EXTRACT_EPI64_0(ctx_ptr->r[reg_idx]))
#define GPR_VEC(ctx_ptr, reg_idx) ((reg_idx == 0) ? _mm_setzero_si128() : ctx_ptr->r[reg_idx])

static inline void Ps2SetGprLow64(R5900Context *ctx, int reg, __m128i new_low)
{
    if (reg != 0)
    {
        ctx->r[reg] = _mm_castpd_si128(_mm_move_sd(_mm_castsi128_pd(ctx->r[reg]), _mm_castsi128_pd(new_low)));
    }
}

#define SET_GPR_U32(ctx_ptr, reg_idx, val)                                \
    do                                                                    \
    {                                                                     \
        if ((reg_idx) != 0)                                               \
        {                                                                 \
            __m128i _newVal = _mm_cvtsi64_si128((int64_t)(int32_t)(val)); \
                                                                          \
            Ps2SetGprLow64(ctx_ptr, reg_idx, _newVal);                    \
        }                                                                 \
    } while (0)


#define SET_GPR_ZE32(ctx_ptr, reg_idx, val)                               \
    do                                                                    \
    {                                                                     \
        if ((reg_idx) != 0)                                               \
        {                                                                 \
            __m128i _newVal = _mm_cvtsi64_si128((int64_t)(uint32_t)(val)); \
            Ps2SetGprLow64(ctx_ptr, reg_idx, _newVal);                    \
        }                                                                 \
    } while (0)

#define SET_GPR_S32(ctx_ptr, reg_idx, val)                                \
    do                                                                    \
    {                                                                     \
        if ((reg_idx) != 0)                                               \
        {                                                                 \
            __m128i _newVal = _mm_cvtsi64_si128((int64_t)(int32_t)(val)); \
            Ps2SetGprLow64(ctx_ptr, reg_idx, _newVal);                    \
        }                                                                 \
    } while (0)

#define SET_GPR_U64(ctx_ptr, reg_idx, val)                       \
    do                                                           \
    {                                                            \
        if ((reg_idx) != 0)                                      \
        {                                                        \
            __m128i _newVal = _mm_cvtsi64_si128((int64_t)(val)); \
            Ps2SetGprLow64(ctx_ptr, reg_idx, _newVal);           \
        }                                                        \
    } while (0)

#define SET_GPR_S64(ctx_ptr, reg_idx, val) SET_GPR_U64(ctx_ptr, reg_idx, val)

#define SET_GPR_VEC(ctx_ptr, reg_idx, val) \
    do                                     \
    {                                      \
        if (reg_idx != 0)                  \
            ctx_ptr->r[reg_idx] = (val);   \
    } while (0)

#endif // PS2_RUNTIME_MACROS_H
