// Optimised software rasterizer paths for GSCpuBackend.
//
// Included from gs_cpu_backend.cpp (after the reference helpers it reuses). Every function here
// must produce results bit-identical to the reference DrawTriangle / DrawSprite / WritePixel /
// SampleTexture paths, including the order of texture page cache accesses and VRAM writes.
// The gains come from hoisting per-primitive state decoding out of the pixel loop, exact
// per-row span computation instead of testing the whole bounding box, compile-time pixel
// formats for frame / depth / texture access, and a cached decoded CLUT.
// Verified with the GS capture/replay harness (see gs_capture.h).

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define PS2X_GS_SIMD_SPAN 1
#endif

namespace gs_fast_detail
{
    inline int64_t floorDiv64(int64_t n, int64_t d)
    {
        // d > 0
        int64_t q = n / d;
        if ((n % d) != 0 && n < 0)
            --q;
        return q;
    }

    inline int64_t ceilDiv64(int64_t n, int64_t d)
    {
        return -floorDiv64(-n, d);
    }

    template <uint8_t PSM>
    struct FastPsm;

#define PS2X_FAST_PSM(gsPsm, memMode, tableName)                                              \
    template <>                                                                               \
    struct FastPsm<gsPsm>                                                                     \
    {                                                                                         \
        static constexpr GSMem::PixelStorageMode kMode = memMode;                             \
        using Traits = GSMem::PixelStorageTraits<memMode>;                                    \
        using PackedT = typename Traits::PackedT;                                             \
        static const typename Traits::PageLookupTableT &Table() { return GSMem::tableName; } \
    };

    PS2X_FAST_PSM(GS_PSM_CT32, GSMem::C32, PageTableC32)
    PS2X_FAST_PSM(GS_PSM_CT24, GSMem::C24, PageTableC32)
    PS2X_FAST_PSM(GS_PSM_CT16, GSMem::C16, PageTableC16)
    PS2X_FAST_PSM(GS_PSM_CT16S, GSMem::C16S, PageTableC16S)
    PS2X_FAST_PSM(GS_PSM_T8, GSMem::P8, PageTableP8)
    PS2X_FAST_PSM(GS_PSM_T4, GSMem::P4, PageTableP4)
    PS2X_FAST_PSM(GS_PSM_T8H, GSMem::P8H, PageTableC32)
    PS2X_FAST_PSM(GS_PSM_T4HL, GSMem::P4HL, PageTableC32)
    PS2X_FAST_PSM(GS_PSM_T4HH, GSMem::P4HH, PageTableC32)
    PS2X_FAST_PSM(GS_PSM_Z32, GSMem::Z32, PageTableZ32)
    PS2X_FAST_PSM(GS_PSM_Z24, GSMem::Z24, PageTableZ32)
    PS2X_FAST_PSM(GS_PSM_Z16, GSMem::Z16, PageTableZ16)
    PS2X_FAST_PSM(GS_PSM_Z16S, GSMem::Z16S, PageTableZ16S)
#undef PS2X_FAST_PSM

    template <uint8_t PSM>
    inline uint32_t fastExtract(const uint8_t *src, uint32_t shift)
    {
        using P = FastPsm<PSM>;
        typename P::PackedT v;
        std::memcpy(&v, src, sizeof(v));
        switch (P::kMode)
        {
        case GSMem::C24:
        case GSMem::Z24:
            return static_cast<uint32_t>(v) & 0x00FFFFFFu;
        case GSMem::P4:
        case GSMem::P4HL:
        case GSMem::P4HH:
            return (static_cast<uint32_t>(v) >> shift) & 0x0Fu;
        default:
            return static_cast<uint32_t>(v);
        }
    }

    // Direct (uncached) read, same as m_readVramFuncs[PSM]. Returns the byte address too.
    template <uint8_t PSM>
    inline uint32_t fastReadVram(const uint8_t *vram, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y, uint32_t &addr)
    {
        using P = FastPsm<PSM>;
        uint32_t shift;
        addr = GSMem::PixelByteAddress<P::kMode>(P::Table(), bp, bw, x, y, shift);
        return fastExtract<PSM>(vram + addr, shift);
    }

    template <uint8_t PSM>
    inline uint32_t fastAddress(uint32_t bp, uint32_t bw, uint32_t x, uint32_t y)
    {
        using P = FastPsm<PSM>;
        uint32_t shift;
        return GSMem::PixelByteAddress<P::kMode>(P::Table(), bp, bw, x, y, shift);
    }

    // Store at a byte address computed by fastAddress (frame / depth formats only).
    template <uint8_t PSM>
    inline void fastStore(uint8_t *vram, uint32_t addr, uint32_t value)
    {
        using P = FastPsm<PSM>;
        static_assert(P::kMode != GSMem::P4 && P::kMode != GSMem::P4HL && P::kMode != GSMem::P4HH, "no 4-bit stores");
        typename P::PackedT packed = static_cast<typename P::PackedT>(value);
        if constexpr (P::kMode == GSMem::C24 || P::kMode == GSMem::Z24)
        {
            typename P::PackedT old;
            std::memcpy(&old, vram + addr, sizeof(old));
            packed = (old & 0xFF000000u) | (packed & 0x00FFFFFFu);
        }
        std::memcpy(vram + addr, &packed, sizeof(packed));
    }

    struct FastPixelState
    {
        uint8_t *vram = nullptr;
        GSMem::TexturePageCache *cache = nullptr;
        const GSDrawState *state = nullptr;
        uint32_t fbp = 0, fbw = 1, zbp = 0;
        int scanSkip = -1;
        bool fge = false;
        uint8_t fogR = 0, fogG = 0, fogB = 0;
        bool ate = false;
        uint8_t atst = 0, aref = 0, afail = 0;
        bool depthWritable = false;
        uint32_t ztst = 1;
        uint32_t zMax = 0xFFFFFFFFu;
        bool zAlways = false;
        bool abe = false, pabe = false;
        uint8_t asel = 0, bsel = 0, csel = 0, dsel = 0;
        int fix = 0;
        // Selector masks (all ones / zero) for the blend: A/B/D pick Cs, Cd or 0; C picks As, Ad or FIX.
        int aS = 0, aD = 0, bS = 0, bD = 0, dS = 0, dD = 0;
        int cAs = 0, cAd = 0, cFix = 0;
        bool date = false, datm = false;
        bool dither = false;
        uint64_t dimx = 0;
        bool colclamp = false;
        bool fbaOr = false;
        uint32_t fbmsk = 0;
    };

    inline bool fastAlphaPass(uint8_t atst, uint8_t aref, uint8_t alpha)
    {
        switch (atst)
        {
        case 0:
            return false;
        case 1:
            return true;
        case 2:
            return alpha < aref;
        case 3:
            return alpha <= aref;
        case 4:
            return alpha == aref;
        case 5:
            return alpha >= aref;
        case 6:
            return alpha > aref;
        default:
            return alpha != aref;
        }
    }

    constexpr uint8_t kZNone = 0xFFu;

    // Mirrors GSCpuBackend::WritePixel for FPSM in {CT32, CT24, CT16, CT16S} and a depth buffer in
    // {Z32, Z24, Z16, Z16S} (or kZNone when the depth buffer is neither read nor written).
    // The caller guarantees the scissor test and that DATE is not combined with a CT24 frame.
    template <uint8_t FPSM, uint8_t ZPSM>
    struct FastWriter
    {
        static constexpr bool kFb16 = FPSM == GS_PSM_CT16 || FPSM == GS_PSM_CT16S;

        // Prefetch the frame / depth cache lines of a row span (pure hint, no semantic effect).
        static inline void PrefetchSpan(const FastPixelState &p, int x0, int x1, int y)
        {
            for (int x = x0 & ~7; x <= x1; x += 8)
            {
                __builtin_prefetch(p.vram + fastAddress<FPSM>(p.fbp, p.fbw, static_cast<uint32_t>(x), static_cast<uint32_t>(y)), 1);
                if constexpr (ZPSM != kZNone)
                    __builtin_prefetch(p.vram + fastAddress<ZPSM>(p.zbp, p.fbw, static_cast<uint32_t>(x), static_cast<uint32_t>(y)), 1);
            }
        }

        __attribute__((always_inline)) static inline void Write(const FastPixelState &p, int x, int y, uint32_t z,
                                 uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
        {
            if (p.scanSkip >= 0 && static_cast<int>(static_cast<uint32_t>(y) & 1u) == p.scanSkip)
                return;

            if (p.fge)
            {
                const uint32_t inverseFog = 255u - fog;
                r = static_cast<uint8_t>(((static_cast<uint32_t>(fog) * r) >> 8) + ((inverseFog * p.fogR) >> 8));
                g = static_cast<uint8_t>(((static_cast<uint32_t>(fog) * g) >> 8) + ((inverseFog * p.fogG) >> 8));
                b = static_cast<uint8_t>(((static_cast<uint32_t>(fog) * b) >> 8) + ((inverseFog * p.fogB) >> 8));
            }

            bool writeRgb = true, writeAlpha = true, writeDepth = true;
            if (p.ate && !fastAlphaPass(p.atst, p.aref, a))
            {
                switch (p.afail)
                {
                case 1:
                    writeDepth = false;
                    break;
                case 2:
                    writeRgb = false;
                    writeAlpha = false;
                    break;
                case 3:
                    if (FPSM == GS_PSM_CT32)
                    {
                        writeAlpha = false;
                        writeDepth = false;
                    }
                    else
                        writeDepth = false;
                    break;
                default:
                    writeRgb = writeAlpha = writeDepth = false;
                    break;
                }
            }
            if (!p.depthWritable)
                writeDepth = false;
            const bool writesFb = writeRgb || writeAlpha;
            if (!writesFb && !writeDepth)
                return;

            const bool preserveDestinationAlpha = FPSM == GS_PSM_CT32 && writeRgb && !writeAlpha;
            // DATE needs the destination for CT32/CT16/CT16S (CT24 + DATE never reaches here).
            const bool frmw = p.date || (writesFb && (p.fbmsk != 0u || p.abe || preserveDestinationAlpha));

            uint8_t *const vram = p.vram;
            uint32_t faddr = 0;
            uint32_t rawFramebufferPixel = 0;
            uint32_t fbrgba = 0;
            if (frmw)
            {
                rawFramebufferPixel = fastReadVram<FPSM>(vram, p.fbp, p.fbw, static_cast<uint32_t>(x), static_cast<uint32_t>(y), faddr);
                fbrgba = rawFramebufferPixel;
                if constexpr (kFb16)
                    fbrgba = Rgba5551ToRgba8888(static_cast<u16>(fbrgba));
                else if constexpr (FPSM == GS_PSM_CT24)
                    fbrgba |= 0x80000000u;
            }
            else
                faddr = fastAddress<FPSM>(p.fbp, p.fbw, static_cast<uint32_t>(x), static_cast<uint32_t>(y));

            if (p.date)
            {
                const uint32_t bit = kFb16 ? ((rawFramebufferPixel >> 15) & 0x1u) : ((rawFramebufferPixel >> 31) & 0x1u);
                if ((bit != 0u) != p.datm)
                    return;
            }

            if (z > p.zMax)
                z = p.zMax;

            uint32_t zaddr = 0;
            bool zpass;
            if constexpr (ZPSM == kZNone)
            {
                zpass = p.ztst == 1u;
            }
            else
            {
                if (p.ztst >= 2u)
                {
                    const uint32_t zb = fastReadVram<ZPSM>(vram, p.zbp, p.fbw, static_cast<uint32_t>(x), static_cast<uint32_t>(y), zaddr);
                    zpass = p.ztst == 2u ? z >= zb : z > zb;
                }
                else
                {
                    zpass = p.ztst == 1u;
                    if (writeDepth)
                        zaddr = fastAddress<ZPSM>(p.zbp, p.fbw, static_cast<uint32_t>(x), static_cast<uint32_t>(y));
                }
            }
            if (!zpass && !p.zAlways)
                return;

            if (writesFb)
            {
                int ir = r;
                int ig = g;
                int ib = b;

                if (p.abe && !(p.pabe && (a & 0x80u) == 0u))
                {
                    const int dr = static_cast<int>(fbrgba & 0xFFu);
                    const int dg = static_cast<int>((fbrgba >> 8) & 0xFFu);
                    const int db = static_cast<int>((fbrgba >> 16) & 0xFFu);
                    const int da = static_cast<int>((fbrgba >> 24) & 0xFFu);
                    // Branch-free selector: sel 0 -> Cs, 1 -> Cd, 2/3 -> 0 (masks from SetupPixelState).
                    const int cAlpha = (a & p.cAs) | (da & p.cAd) | p.cFix;
                    ir = ((((ir & p.aS) | (dr & p.aD)) - ((ir & p.bS) | (dr & p.bD))) * cAlpha >> 7) + ((ir & p.dS) | (dr & p.dD));
                    ig = ((((ig & p.aS) | (dg & p.aD)) - ((ig & p.bS) | (dg & p.bD))) * cAlpha >> 7) + ((ig & p.dS) | (dg & p.dD));
                    ib = ((((ib & p.aS) | (db & p.aD)) - ((ib & p.bS) | (db & p.bD))) * cAlpha >> 7) + ((ib & p.dS) | (db & p.dD));
                }

                if (kFb16 && p.dither)
                {
                    const uint32_t shift = ((static_cast<uint32_t>(y) & 3u) * 16u) + ((static_cast<uint32_t>(x) & 3u) * 4u);
                    int dm = static_cast<int>((p.dimx >> shift) & 0x7u);
                    if (dm & 4)
                        dm -= 8;
                    ir += dm;
                    ig += dm;
                    ib += dm;
                }

                if (p.colclamp)
                {
                    ir = clampU8(ir);
                    ig = clampU8(ig);
                    ib = clampU8(ib);
                }
                else
                {
                    ir &= 0xFF;
                    ig &= 0xFF;
                    ib &= 0xFF;
                }

                uint8_t oa = a;
                if (writeAlpha && p.fbaOr)
                    oa = static_cast<uint8_t>(oa | 0x80u);

                u32 pixel = pack32(static_cast<u8>(ir), static_cast<u8>(ig), static_cast<u8>(ib), oa);
                if (p.fbmsk != 0u)
                    pixel = (pixel & ~p.fbmsk) | (fbrgba & p.fbmsk);
                if (preserveDestinationAlpha)
                    pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
                if constexpr (kFb16)
                    pixel = Rgba8888ToRgba5551(pixel);

                p.cache->NoteWrite(vram, faddr);
                fastStore<FPSM>(vram, faddr, pixel);
            }

            if constexpr (ZPSM != kZNone)
            {
                if (writeDepth)
                {
                    p.cache->NoteWrite(vram, zaddr);
                    fastStore<ZPSM>(vram, zaddr, z);
                }
            }
        }
    };

    struct FastLevel
    {
        uint32_t tbp = 0, tbw = 0;
        int texW = 1, texH = 1;
        uint16_t minU = 0, maxU = 0, minV = 0, maxV = 0;
    };

    struct FastSampler
    {
        const uint8_t *vram = nullptr;
        GSMem::TexturePageCache *cache = nullptr;
        FastLevel lv[7];
        uint8_t wrapU = 0, wrapV = 0;
        const uint32_t *palette = nullptr;
        GSTexaReg texa{};
        GSCpuBackend *self = nullptr;
        const GSDrawState *state = nullptr;
        // TEX1 LOD parameters (MXL != 0)
        bool lodFromQ = false; // !LCM && !FST
        uint32_t l = 0;
        int32_t k = 0;
        uint32_t mxl = 0, mmin = 0;
        bool mmag = false;
    };

    template <uint8_t TPSM>
    inline uint32_t fastFetchTexel(const FastSampler &s, const FastLevel &lv, int u, int v)
    {
        using P = FastPsm<TPSM>;
        uint32_t shift;
        const uint32_t addr = GSMem::PixelByteAddress<P::kMode>(P::Table(), lv.tbp, lv.tbw,
                                                                 static_cast<uint32_t>(u), static_cast<uint32_t>(v), shift);
        const uint32_t out = fastExtract<TPSM>(s.cache->Resolve(s.vram, addr), shift);
        if constexpr (TPSM == GS_PSM_CT32 || TPSM == GS_PSM_Z32)
            return out;
        else if constexpr (TPSM == GS_PSM_CT24 || TPSM == GS_PSM_Z24)
            return applyTexa(s.texa, GS_PSM_CT24, out);
        else if constexpr (TPSM == GS_PSM_CT16 || TPSM == GS_PSM_CT16S || TPSM == GS_PSM_Z16 || TPSM == GS_PSM_Z16S)
            return applyTexa(s.texa, GS_PSM_CT16, Rgba5551ToRgba8888(static_cast<u16>(out)));
        else
            return s.palette[static_cast<u8>(out)];
    }

    // One texture level (sampleLevel in SampleTexture); uu/vv already shifted by the level.
    template <uint8_t TPSM, bool LINEAR>
    inline uint32_t fastSampleOneLevel(const FastSampler &s, const FastLevel &lv, int32_t uu, int32_t vv)
    {
        if (!LINEAR)
        {
            const int su = wrapTextureCoordinate(uu >> 16, lv.texW, s.wrapU, lv.minU, lv.maxU);
            const int sv = wrapTextureCoordinate(vv >> 16, lv.texH, s.wrapV, lv.minV, lv.maxV);
            return fastFetchTexel<TPSM>(s, lv, su, sv);
        }
        uu -= 0x8000;
        vv -= 0x8000;
        const int fu = (uu >> 12) & 0xF;
        const int fv = (vv >> 12) & 0xF;
        const int u0 = wrapTextureCoordinate(uu >> 16, lv.texW, s.wrapU, lv.minU, lv.maxU);
        const int u1 = wrapTextureCoordinate((uu >> 16) + 1, lv.texW, s.wrapU, lv.minU, lv.maxU);
        const int v0 = wrapTextureCoordinate(vv >> 16, lv.texH, s.wrapV, lv.minV, lv.maxV);
        const int v1 = wrapTextureCoordinate((vv >> 16) + 1, lv.texH, s.wrapV, lv.minV, lv.maxV);
        const uint32_t c00 = fastFetchTexel<TPSM>(s, lv, u0, v0);
        const uint32_t c01 = fastFetchTexel<TPSM>(s, lv, u1, v0);
        const uint32_t c10 = fastFetchTexel<TPSM>(s, lv, u0, v1);
        const uint32_t c11 = fastFetchTexel<TPSM>(s, lv, u1, v1);
        uint32_t result = 0u;
        for (uint32_t sh = 0; sh < 32u; sh += 8u)
        {
            const int top = lerpTexel4(static_cast<int>((c00 >> sh) & 0xFFu), static_cast<int>((c01 >> sh) & 0xFFu), fu);
            const int bottom = lerpTexel4(static_cast<int>((c10 >> sh) & 0xFFu), static_cast<int>((c11 >> sh) & 0xFFu), fu);
            result |= static_cast<uint32_t>(lerpTexel4(top, bottom, fv) & 0xFF) << sh;
        }
        return result;
    }

    // SampleTexture for MXL == 0 (level 0 only, filter from state.linearFilter).
    template <uint8_t TPSM, bool LINEAR>
    uint32_t fastSampleLevel0(const FastSampler &s, int32_t uu, int32_t vv, float)
    {
        return fastSampleOneLevel<TPSM, LINEAR>(s, s.lv[0], uu, vv);
    }

    // SampleTexture for MXL != 0: per-pixel LOD exactly as the reference.
    template <uint8_t TPSM>
    uint32_t fastSampleMip(const FastSampler &s, int32_t u, int32_t v, float q)
    {
        const uint32_t mxl = s.mxl;
        const bool mmag = s.mmag;
        const uint32_t mmin = s.mmin;
        int lodi = 0;
        int32_t lodf = 0;
        bool trilinear = false;
        bool linear;
        {
            const uint32_t l = s.l;
            int32_t k = s.k;
            double lod = static_cast<double>(k) / 16.0;
            if (s.lodFromQ)
            {
                const double aq = std::max<double>(std::fabs(static_cast<double>(q)), 1.0e-30);
                lod += -std::log2(aq) * static_cast<double>(1u << l);
            }
            const bool minLinear = mmin == 1u || mmin == 4u || mmin == 5u;
            linear = lod > 0.0 ? minLinear : mmag;
            if (lod > 0.0 && mmin >= 2u && mmin <= 5u)
            {
                const int maxLevel = static_cast<int>(std::min<uint32_t>(mxl, 6u));
                int32_t lodFixed = static_cast<int32_t>(std::min(lod, static_cast<double>(maxLevel)) * 65536.0);
                if (mmin == 3u || mmin == 5u)
                {
                    lodi = lodFixed >> 16;
                    lodf = lodFixed & 0xFFFF;
                    trilinear = lodi < maxLevel && lodf != 0;
                }
                else
                {
                    lodFixed += 0x8000;
                    lodi = std::min(lodFixed >> 16, maxLevel);
                }
            }
        }

        const FastLevel &lv = s.lv[lodi];
        const uint32_t c = linear ? fastSampleOneLevel<TPSM, true>(s, lv, u >> lodi, v >> lodi)
                                  : fastSampleOneLevel<TPSM, false>(s, lv, u >> lodi, v >> lodi);
        if (!trilinear)
            return c;

        const int level2 = lodi + 1;
        const FastLevel &lv2 = s.lv[level2];
        const uint32_t c2 = linear ? fastSampleOneLevel<TPSM, true>(s, lv2, u >> level2, v >> level2)
                                   : fastSampleOneLevel<TPSM, false>(s, lv2, u >> level2, v >> level2);
        const int f15 = lodf >> 1;
        uint32_t result = 0u;
        for (uint32_t sh = 0; sh < 32u; sh += 8u)
        {
            const int a0 = static_cast<int>((c >> sh) & 0xFFu);
            const int a1 = static_cast<int>((c2 >> sh) & 0xFFu);
            result |= static_cast<uint32_t>((a0 + (((a1 - a0) * f15) >> 15)) & 0xFF) << sh;
        }
        return result;
    }

    using FastSampleFn = uint32_t (*)(const FastSampler &, int32_t, int32_t, float);

    // combineTexture (reference) inlined into the pixel loops; texel is packed RGBA8888.
    __attribute__((always_inline)) inline TextureCombineResult fastCombine(const GSTex0Reg &tex, uint8_t vr, uint8_t vg, uint8_t vb,
                                                                          uint8_t va, uint32_t texel)
    {
        const uint8_t tr = static_cast<uint8_t>(texel);
        const uint8_t tg = static_cast<uint8_t>(texel >> 8);
        const uint8_t tb = static_cast<uint8_t>(texel >> 16);
        const uint8_t ta = static_cast<uint8_t>(texel >> 24);
        const bool textureHasAlpha = tex.tcc != 0u;
        TextureCombineResult out{tr, tg, tb, textureHasAlpha ? ta : va};
        switch (tex.tfx)
        {
        case 0: // MODULATE
            out.r = clampU8((tr * vr) >> 7);
            out.g = clampU8((tg * vg) >> 7);
            out.b = clampU8((tb * vb) >> 7);
            out.a = textureHasAlpha ? clampU8((ta * va) >> 7) : va;
            break;
        case 2: // HIGHLIGHT
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? clampU8(ta + va) : va;
            break;
        case 3: // HIGHLIGHT2
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? ta : va;
            break;
        default: // DECAL
            break;
        }
        return out;
    }

    // Row interleaving for parallel rasterization: rows are split into 2^shift-row bands and
    // band b is drawn by thread b % count.
    struct FastBand
    {
        uint32_t shift = 0;
        uint32_t count = 1;
        uint32_t index = 0;
        bool Has(int y) const { return count == 1u || ((static_cast<uint32_t>(y) >> shift) % count) == index; }
        // First owned row >= y (y >= 0).
        int NextRow(int y) const
        {
            if (count == 1u)
                return y;
            const uint32_t b = static_cast<uint32_t>(y) >> shift;
            const uint32_t add = (index + count - b % count) % count;
            return add == 0u ? y : static_cast<int>((b + add) << shift);
        }
        // Last row of the run of consecutive owned rows containing the owned row y.
        int RunEnd(int y) const
        {
            if (count == 1u)
                return INT32_MAX;
            return static_cast<int>(((static_cast<uint32_t>(y) >> shift) + 1u) << shift) - 1;
        }
        // Whether any row of [y0, y1] is owned (y0 >= 0).
        bool Overlaps(int y0, int y1) const
        {
            return y0 <= y1 && NextRow(y0) <= y1;
        }
    };

    // Last textured pixel drawn by a thread (raster order), to reproduce the serial final
    // texture page cache state after a parallel batch.
    struct FastTrack
    {
        bool any = false;
        int y = 0;
        int x = 0;
        void Row(int y_, int x_)
        {
            any = true;
            y = y_;
            x = x_;
        }
    };

    inline FastSampleFn selectMipSampler(uint8_t psm)
    {
        switch (psm)
        {
        case GS_PSM_CT32: return &fastSampleMip<GS_PSM_CT32>;
        case GS_PSM_CT24: return &fastSampleMip<GS_PSM_CT24>;
        case GS_PSM_CT16: return &fastSampleMip<GS_PSM_CT16>;
        case GS_PSM_CT16S: return &fastSampleMip<GS_PSM_CT16S>;
        case GS_PSM_T8: return &fastSampleMip<GS_PSM_T8>;
        case GS_PSM_T4: return &fastSampleMip<GS_PSM_T4>;
        case GS_PSM_T8H: return &fastSampleMip<GS_PSM_T8H>;
        case GS_PSM_T4HL: return &fastSampleMip<GS_PSM_T4HL>;
        case GS_PSM_T4HH: return &fastSampleMip<GS_PSM_T4HH>;
        case GS_PSM_Z32: return &fastSampleMip<GS_PSM_Z32>;
        case GS_PSM_Z24: return &fastSampleMip<GS_PSM_Z24>;
        case GS_PSM_Z16: return &fastSampleMip<GS_PSM_Z16>;
        case GS_PSM_Z16S: return &fastSampleMip<GS_PSM_Z16S>;
        default: return nullptr;
        }
    }

    template <bool LINEAR>
    FastSampleFn selectLevel0Sampler(uint8_t psm)
    {
        switch (psm)
        {
        case GS_PSM_CT32: return &fastSampleLevel0<GS_PSM_CT32, LINEAR>;
        case GS_PSM_CT24: return &fastSampleLevel0<GS_PSM_CT24, LINEAR>;
        case GS_PSM_CT16: return &fastSampleLevel0<GS_PSM_CT16, LINEAR>;
        case GS_PSM_CT16S: return &fastSampleLevel0<GS_PSM_CT16S, LINEAR>;
        case GS_PSM_T8: return &fastSampleLevel0<GS_PSM_T8, LINEAR>;
        case GS_PSM_T4: return &fastSampleLevel0<GS_PSM_T4, LINEAR>;
        case GS_PSM_T8H: return &fastSampleLevel0<GS_PSM_T8H, LINEAR>;
        case GS_PSM_T4HL: return &fastSampleLevel0<GS_PSM_T4HL, LINEAR>;
        case GS_PSM_T4HH: return &fastSampleLevel0<GS_PSM_T4HH, LINEAR>;
        case GS_PSM_Z32: return &fastSampleLevel0<GS_PSM_Z32, LINEAR>;
        case GS_PSM_Z24: return &fastSampleLevel0<GS_PSM_Z24, LINEAR>;
        case GS_PSM_Z16: return &fastSampleLevel0<GS_PSM_Z16, LINEAR>;
        case GS_PSM_Z16S: return &fastSampleLevel0<GS_PSM_Z16S, LINEAR>;
        default: return nullptr;
        }
    }

    // Per-pixel triangle attributes computed for a whole span chunk with 4-wide double SIMD.
    // Every lane performs exactly the operations of the scalar per-pixel code in RasterTriangle:
    //   b1 = double(w1) * invArea, b2 = double(w2) * invArea,
    //   interpolate3(a0, a1, a2, b1, b2) = a0 + (a1 - a0) * b1 + (a2 - a0) * b2, which GCC contracts
    //   (-ffp-contract=fast, FMA available) to fma(a2 - a0, b2, fma(a1 - a0, b1, a0)),
    // followed by the same floor / truncate / clamp / divide steps. Whether the scalar build really
    // contracts that way is checked at runtime against the scalar helpers (SpanSimdUsable); when it
    // does not, the scalar loop is used.
    enum TriChannel
    {
        kChZ = 0,
        kChR,
        kChG,
        kChB,
        kChA,
        kChU, // u (FST) or s
        kChV, // v (FST) or t
        kChQ,
        kChFog,
        kChCount
    };

    struct TriAttrs
    {
        double invArea = 0.0;
        double base[kChCount]{}, d1[kChCount]{}, d2[kChCount]{};
        double texW = 1.0, texH = 1.0;
        bool iip = false, tme = false, fst = false, fge = false;

        void Set(int c, double a0, double a1, double a2)
        {
            base[c] = a0;
            d1[c] = a1 - a0;
            d2[c] = a2 - a0;
        }
    };

    constexpr int kSpanChunk = 64;

    struct alignas(32) SpanAttrs
    {
        uint32_t z[kSpanChunk];
        int32_t r[kSpanChunk], g[kSpanChunk], b[kSpanChunk], a[kSpanChunk];
        int32_t uf[kSpanChunk], vf[kSpanChunk];
        float q[kSpanChunk];
        int32_t fog[kSpanChunk];
    };

    // Scalar per-pixel attributes of vertices v[0..2], written exactly as the RasterTriangle
    // per-pixel code (used to validate the SIMD path at startup).
    inline void PixelAttrsScalar(const TriAttrs &A, const double (&v)[3][kChCount], int64_t w1, int64_t w2, uint32_t &z,
                                 int32_t rgba[4], int32_t &uf, int32_t &vf, float &q, int32_t &fog)
    {
        const double b1 = static_cast<double>(w1) * A.invArea;
        const double b2 = static_cast<double>(w2) * A.invArea;
        const double *v0 = v[0], *v1 = v[1], *v2 = v[2];
        z = vertexZ(std::floor(interpolate3(v0[kChZ], v1[kChZ], v2[kChZ], b1, b2) + 1.0e-6));
        for (int c = 0; c < 4; ++c)
            rgba[c] = clampU8(truncateAttribute(interpolate3(v0[kChR + c], v1[kChR + c], v2[kChR + c], b1, b2)));
        q = 1.0f;
        if (A.fst)
        {
            uf = textureCoordToFixed(interpolate3(v0[kChU], v1[kChU], v2[kChU], b1, b2) / 16.0);
            vf = textureCoordToFixed(interpolate3(v0[kChV], v1[kChV], v2[kChV], b1, b2) / 16.0);
        }
        else
        {
            const double is = interpolate3(v0[kChU], v1[kChU], v2[kChU], b1, b2);
            const double it = interpolate3(v0[kChV], v1[kChV], v2[kChV], b1, b2);
            double iq = interpolate3(v0[kChQ], v1[kChQ], v2[kChQ], b1, b2);
            if (iq == 0.0)
                iq = 1.0e-30;
            uf = textureCoordToFixed(is / iq * A.texW);
            vf = textureCoordToFixed(it / iq * A.texH);
            q = static_cast<float>(iq);
        }
        fog = clampU8(truncateAttribute(interpolate3(v0[kChFog], v1[kChFog], v2[kChFog], b1, b2)));
    }

#if defined(PS2X_GS_SIMD_SPAN)
    __attribute__((always_inline)) inline __m256d spanInterp(const TriAttrs &A, int c, __m256d b1, __m256d b2)
    {
        return _mm256_fmadd_pd(_mm256_set1_pd(A.d2[c]), b2,
                               _mm256_fmadd_pd(_mm256_set1_pd(A.d1[c]), b1, _mm256_set1_pd(A.base[c])));
    }

    // clampU8(truncateAttribute(v)) as int32 lanes.
    __attribute__((always_inline)) inline __m128i spanTruncClamp(__m256d v)
    {
        const __m256d f = _mm256_round_pd(_mm256_add_pd(v, _mm256_set1_pd(1.0e-6)), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        const __m128i i = _mm256_cvttpd_epi32(f);
        return _mm_min_epi32(_mm_max_epi32(i, _mm_setzero_si128()), _mm_set1_epi32(255));
    }

    // textureCoordToFixed.
    __attribute__((always_inline)) inline __m128i spanTexFixed(__m256d t)
    {
        const __m256d nan = _mm256_cmp_pd(t, t, _CMP_UNORD_Q);
        t = _mm256_andnot_pd(nan, t);
        t = _mm256_max_pd(t, _mm256_set1_pd(-32766.0));
        t = _mm256_min_pd(t, _mm256_set1_pd(32766.0));
        return _mm256_cvttpd_epi32(_mm256_mul_pd(t, _mm256_set1_pd(65536.0)));
    }

    // Low 32 bits of each 64-bit lane.
    __attribute__((always_inline)) inline __m128i spanLow32(__m256i v)
    {
        const __m256i p = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6));
        return _mm256_castsi256_si128(p);
    }

    // vertexZ(floor(interpolate3(z) + 1e-6)).
    __attribute__((always_inline)) inline __m128i spanZ(__m256d zi)
    {
        const __m256d zf = _mm256_round_pd(_mm256_add_pd(zi, _mm256_set1_pd(1.0e-6)), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        const __m256d pos = _mm256_cmp_pd(zf, _mm256_setzero_pd(), _CMP_GT_OQ);
        const __m256d big = _mm256_cmp_pd(zf, _mm256_set1_pd(4294967295.0), _CMP_GE_OQ);
        // zf is integral; for 0 < zf < 2^32 adding 2^52 leaves zf in the low mantissa bits.
        const __m256d magic = _mm256_add_pd(zf, _mm256_set1_pd(4503599627370496.0));
        __m128i val = spanLow32(_mm256_castpd_si256(magic));
        const __m128i bigM = spanLow32(_mm256_castpd_si256(big));
        const __m128i posM = spanLow32(_mm256_castpd_si256(pos));
        val = _mm_or_si128(val, bigM);
        return _mm_and_si128(val, posM);
    }

    // Attributes of pixels w = (w1, w2) + i * (s1, s2), i in [0, n) (lanes up to n rounded to 4).
    inline void ComputeSpanAttrs(const TriAttrs &A, double w1, double w2, double s1, double s2, int n, SpanAttrs &o)
    {
        const __m256d lane = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
        __m256d W1 = _mm256_add_pd(_mm256_set1_pd(w1), _mm256_mul_pd(lane, _mm256_set1_pd(s1)));
        __m256d W2 = _mm256_add_pd(_mm256_set1_pd(w2), _mm256_mul_pd(lane, _mm256_set1_pd(s2)));
        const __m256d step1 = _mm256_set1_pd(4.0 * s1);
        const __m256d step2 = _mm256_set1_pd(4.0 * s2);
        const __m256d invA = _mm256_set1_pd(A.invArea);
        for (int i = 0; i < n; i += 4)
        {
            const __m256d b1 = _mm256_mul_pd(W1, invA);
            const __m256d b2 = _mm256_mul_pd(W2, invA);
            _mm_store_si128(reinterpret_cast<__m128i *>(o.z + i), spanZ(spanInterp(A, kChZ, b1, b2)));
            if (A.iip)
            {
                _mm_store_si128(reinterpret_cast<__m128i *>(o.r + i), spanTruncClamp(spanInterp(A, kChR, b1, b2)));
                _mm_store_si128(reinterpret_cast<__m128i *>(o.g + i), spanTruncClamp(spanInterp(A, kChG, b1, b2)));
                _mm_store_si128(reinterpret_cast<__m128i *>(o.b + i), spanTruncClamp(spanInterp(A, kChB, b1, b2)));
                _mm_store_si128(reinterpret_cast<__m128i *>(o.a + i), spanTruncClamp(spanInterp(A, kChA, b1, b2)));
            }
            if (A.tme)
            {
                if (A.fst)
                {
                    const __m256d inv16 = _mm256_set1_pd(0.0625);
                    _mm_store_si128(reinterpret_cast<__m128i *>(o.uf + i), spanTexFixed(_mm256_mul_pd(spanInterp(A, kChU, b1, b2), inv16)));
                    _mm_store_si128(reinterpret_cast<__m128i *>(o.vf + i), spanTexFixed(_mm256_mul_pd(spanInterp(A, kChV, b1, b2), inv16)));
                }
                else
                {
                    const __m256d is = spanInterp(A, kChU, b1, b2);
                    const __m256d it = spanInterp(A, kChV, b1, b2);
                    __m256d iq = spanInterp(A, kChQ, b1, b2);
                    iq = _mm256_blendv_pd(iq, _mm256_set1_pd(1.0e-30), _mm256_cmp_pd(iq, _mm256_setzero_pd(), _CMP_EQ_OQ));
                    _mm_store_si128(reinterpret_cast<__m128i *>(o.uf + i), spanTexFixed(_mm256_mul_pd(_mm256_div_pd(is, iq), _mm256_set1_pd(A.texW))));
                    _mm_store_si128(reinterpret_cast<__m128i *>(o.vf + i), spanTexFixed(_mm256_mul_pd(_mm256_div_pd(it, iq), _mm256_set1_pd(A.texH))));
                    _mm_store_ps(o.q + i, _mm256_cvtpd_ps(iq));
                }
            }
            if (A.fge)
                _mm_store_si128(reinterpret_cast<__m128i *>(o.fog + i), spanTruncClamp(spanInterp(A, kChFog, b1, b2)));
            W1 = _mm256_add_pd(W1, step1);
            W2 = _mm256_add_pd(W2, step2);
        }
    }

    // Startup check that the SIMD lanes reproduce the scalar helpers bit for bit in this build
    // (FMA contraction of interpolate3 included). PS2X_GS_NO_SIMD=1 forces the scalar loop.
    inline bool SpanSimdUsable()
    {
        static const bool usable = []()
        {
            if (const char *e = std::getenv("PS2X_GS_NO_SIMD"); e && e[0] == '1')
                return false;
            auto fail = []()
            {
                std::fprintf(stderr, "[gs] SIMD span attributes disabled: they do not match the scalar path in this build\n");
                return false;
            };
            uint64_t seed = 0x9E3779B97F4A7C15ull;
            auto next = [&]()
            {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                return seed;
            };
            auto unit = [&]()
            { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); };
            for (int iter = 0; iter < 4000; ++iter)
            {
                TriAttrs A;
                const int64_t area = 1 + static_cast<int64_t>(next() % (iter < 2000 ? 4096u : (1u << 30)));
                A.invArea = 1.0 / static_cast<double>(area);
                A.texW = static_cast<double>(1u << (next() % 11u));
                A.texH = static_cast<double>(1u << (next() % 11u));
                A.iip = A.tme = A.fge = true;
                A.fst = (iter & 1) != 0;
                double v[3][kChCount];
                for (int k = 0; k < 3; ++k)
                {
                    v[k][kChZ] = (iter % 3 == 0) ? std::floor(unit() * 4294967296.0) : (unit() * 4.4e9 - 0.1e9);
                    for (int c = kChR; c <= kChA; ++c)
                        v[k][c] = static_cast<double>(next() & 0xFFu);
                    if (A.fst)
                    {
                        v[k][kChU] = static_cast<double>(next() & 0xFFFFu);
                        v[k][kChV] = static_cast<double>(next() & 0xFFFFu);
                    }
                    else
                    {
                        v[k][kChU] = static_cast<double>(static_cast<float>(unit() * 4.0 - 1.0));
                        v[k][kChV] = static_cast<double>(static_cast<float>(unit() * 4.0 - 1.0));
                    }
                    v[k][kChQ] = (iter % 7 == 0) ? 0.0 : static_cast<double>(static_cast<float>(unit() * 2.0 - 0.5));
                    v[k][kChFog] = static_cast<double>(next() & 0xFFu);
                }
                for (int c = 0; c < kChCount; ++c)
                    A.Set(c, v[0][c], v[1][c], v[2][c]);
                const int64_t s1 = static_cast<int64_t>(next() % 4097u) - 2048;
                const int64_t s2 = static_cast<int64_t>(next() % 4097u) - 2048;
                const int64_t w1 = static_cast<int64_t>(next() % static_cast<uint64_t>(3 * area)) - area;
                const int64_t w2 = static_cast<int64_t>(next() % static_cast<uint64_t>(3 * area)) - area;
                SpanAttrs o;
                ComputeSpanAttrs(A, static_cast<double>(w1), static_cast<double>(w2), static_cast<double>(s1),
                                 static_cast<double>(s2), 8, o);
                for (int i = 0; i < 8; ++i)
                {
                    const int64_t pw1 = w1 + s1 * i, pw2 = w2 + s2 * i;
                    uint32_t z;
                    int32_t rgba[4], uf, vf, fog;
                    float q;
                    PixelAttrsScalar(A, v, pw1, pw2, z, rgba, uf, vf, q, fog);
                    if (z != o.z[i] || rgba[0] != o.r[i] || rgba[1] != o.g[i] || rgba[2] != o.b[i] || rgba[3] != o.a[i] ||
                        uf != o.uf[i] || vf != o.vf[i] || fog != o.fog[i] ||
                        (!A.fst && std::memcmp(&q, &o.q[i], sizeof(q)) != 0))
                        return fail();
                    // Raw interpolation must match bit for bit (catches contraction differences that
                    // the rounding steps above would rarely expose).
                    const double b1 = static_cast<double>(pw1) * A.invArea;
                    const double b2 = static_cast<double>(pw2) * A.invArea;
                    const double ref = interpolate3(v[0][kChZ], v[1][kChZ], v[2][kChZ], b1, b2);
                    alignas(32) double lanes[4];
                    _mm256_store_pd(lanes, spanInterp(A, kChZ, _mm256_set1_pd(b1), _mm256_set1_pd(b2)));
                    if (std::memcmp(&ref, &lanes[0], sizeof(ref)) != 0)
                        return fail();
                }
            }
            return true;
        }();
        return usable;
    }
#endif
} // namespace gs_fast_detail
using namespace gs_fast_detail;

struct GSFastRaster
{
    static uint32_t SampleGeneric(const FastSampler &s, int32_t u, int32_t v, float q)
    {
        return s.self->SampleTexture(*s.state, u, v, q);
    }

    static bool SetupPixelState(GSCpuBackend &be, const GSDrawState &state, FastPixelState &p, uint8_t &zKind)
    {
        const auto &ctx = state.context;
        const uint8_t fpsm = ctx.frame.psm;
        if (fpsm != GS_PSM_CT32 && fpsm != GS_PSM_CT24 && fpsm != GS_PSM_CT16 && fpsm != GS_PSM_CT16S)
            return false;
        const uint64_t test = ctx.test;
        p.date = ((test >> 14) & 0x1u) != 0u;
        if (p.date && fpsm == GS_PSM_CT24)
            return false; // draws nothing; leave it to the reference path
        p.datm = ((test >> 15) & 0x1u) != 0u;

        p.vram = be.m_vram;
        p.cache = &be.m_texturePageCache;
        p.state = &state;
        p.fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
        p.fbw = std::max<u32>(ctx.frame.fbw, 1u);
        p.zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
        const uint32_t scanmsk = static_cast<uint32_t>(state.scanmsk & 3u);
        p.scanSkip = scanmsk >= 2u ? static_cast<int>(scanmsk & 1u) : -1;
        p.fge = state.prim.fge;
        p.fogR = state.fogR;
        p.fogG = state.fogG;
        p.fogB = state.fogB;
        p.ate = (test & 0x1u) != 0u;
        p.atst = static_cast<uint8_t>((test >> 1) & 0x7u);
        p.aref = static_cast<uint8_t>((test >> 4) & 0xFFu);
        p.afail = static_cast<uint8_t>((test >> 12) & 0x3u);
        const bool zte = ((test >> 16) & 0x1u) != 0u;
        p.depthWritable = zte && !ctx.zbuf.zmask;
        p.ztst = zte ? static_cast<uint32_t>((test >> 17) & 3u) : 1u;
        const uint32_t zFormat = ctx.zbuf.psm & 0x3u;
        p.zMax = 0xFFFFFFFFu >> (zFormat * 8u);
        static const bool zAlways = std::getenv("PS2X_GS_ZALWAYS") != nullptr;
        p.zAlways = zAlways;
        p.abe = state.prim.abe;
        p.pabe = state.pabe;
        const uint64_t alphaReg = ctx.alpha;
        p.asel = alphaReg & 3;
        p.bsel = (alphaReg >> 2) & 3;
        p.csel = (alphaReg >> 4) & 3;
        p.dsel = (alphaReg >> 6) & 3;
        p.fix = static_cast<int>((alphaReg >> 32) & 0xFF);
        p.aS = p.asel == 0 ? -1 : 0;
        p.aD = p.asel == 1 ? -1 : 0;
        p.bS = p.bsel == 0 ? -1 : 0;
        p.bD = p.bsel == 1 ? -1 : 0;
        p.dS = p.dsel == 0 ? -1 : 0;
        p.dD = p.dsel == 1 ? -1 : 0;
        p.cAs = p.csel == 0 ? -1 : 0;
        p.cAd = p.csel == 1 ? -1 : 0;
        p.cFix = p.csel >= 2 ? p.fix : 0;
        p.dither = (state.dthe & 1u) != 0u;
        p.dimx = state.dimx;
        p.colclamp = (state.colclamp & 1u) != 0u;
        p.fbaOr = (ctx.fba & 0x1ull) != 0ull && fpsm != GS_PSM_CT24;
        p.fbmsk = ctx.frame.fbmsk;

        const bool zAccess = p.ztst >= 2u || p.depthWritable;
        if (!zAccess)
            zKind = kZNone;
        else
        {
            zKind = ctx.zbuf.psm;
            if (zKind != GS_PSM_Z32 && zKind != GS_PSM_Z24 && zKind != GS_PSM_Z16 && zKind != GS_PSM_Z16S)
                return false;
        }
        return true;
    }

    static FastSampleFn SetupSampler(GSCpuBackend &be, const GSDrawState &state, FastSampler &s)
    {
        const auto &ctx = state.context;
        const auto &tex = ctx.tex0;
        s.vram = be.m_vram;
        s.cache = &be.m_texturePageCache;
        s.self = &be;
        s.state = &state;
        const uint64_t tex1 = ctx.tex1;
        const uint32_t mxl = static_cast<uint32_t>((tex1 >> 2) & 0x7u);
        const uint64_t clamp = ctx.clamp;
        s.wrapU = static_cast<uint8_t>(clamp & 0x3u);
        s.wrapV = static_cast<uint8_t>((clamp >> 2) & 0x3u);
        const uint32_t levels = mxl != 0u ? 7u : 1u;
        for (uint32_t level = 0; level < levels; ++level)
        {
            FastLevel &lv = s.lv[level];
            uint32_t tbp = tex.tbp0;
            uint32_t tbw = tex.tbw;
            if (level > 0)
            {
                const uint64_t m = level <= 3 ? ctx.miptbp1 : ctx.miptbp2;
                const uint32_t shift = static_cast<uint32_t>(((level - 1) % 3) * 20);
                tbp = static_cast<uint32_t>((m >> shift) & 0x3FFFu);
                tbw = static_cast<uint32_t>((m >> (shift + 14u)) & 0x3Fu);
            }
            lv.tbp = tbp;
            lv.tbw = tbw;
            lv.texW = std::max(1, static_cast<int>(state.textureWidth) >> level);
            lv.texH = std::max(1, static_cast<int>(state.textureHeight) >> level);
            lv.minU = static_cast<uint16_t>(((clamp >> 4) & 0x3FFu) >> level);
            lv.maxU = static_cast<uint16_t>(((clamp >> 14) & 0x3FFu) >> level);
            lv.minV = static_cast<uint16_t>(((clamp >> 24) & 0x3FFu) >> level);
            lv.maxV = static_cast<uint16_t>(((clamp >> 34) & 0x3FFu) >> level);
        }
        s.texa = state.texa;
        if (isFourBitIndexedPsm(tex.psm) || isEightBitIndexedPsm(tex.psm))
            s.palette = be.GetPalette(state);
        if (mxl != 0u)
        {
            s.mxl = mxl;
            s.mmag = ((tex1 >> 5) & 0x1u) != 0u;
            s.mmin = static_cast<uint32_t>((tex1 >> 6) & 0x7u);
            s.lodFromQ = (tex1 & 1u) == 0u && !state.prim.fst;
            s.l = static_cast<uint32_t>((tex1 >> 19) & 0x3u);
            int32_t k = static_cast<int32_t>((tex1 >> 32) & 0xFFFu);
            if (k & 0x800)
                k -= 0x1000;
            s.k = k;
            FastSampleFn fn = selectMipSampler(tex.psm);
            return fn ? fn : &SampleGeneric;
        }
        FastSampleFn fn = state.linearFilter ? selectLevel0Sampler<true>(tex.psm) : selectLevel0Sampler<false>(tex.psm);
        return fn ? fn : &SampleGeneric;
    }

    template <class W>
    static void RasterTriangle(const GSPrimitiveBatch &batch, const FastPixelState &psIn, const FastSampler &smp, FastSampleFn sample, const FastBand &band, FastTrack *track)
    {
        // Local copy: the per-pixel sampler call cannot modify it, so state loads and the row-
        // invariant address math can stay hoisted out of the pixel loop.
        const FastPixelState ps = psIn;
        const GSDrawState &state = batch.state;
        const auto &ctx = state.context;
        const GSVertex *vtx[3] = {&batch.vertices[0], &batch.vertices[1], &batch.vertices[2]};

        int64_t X[3], Y[3];
        for (int i = 0; i < 3; ++i)
        {
            X[i] = vertexFixedX(*vtx[i], ctx);
            Y[i] = vertexFixedY(*vtx[i], ctx);
        }

        const int64_t area2 = (X[1] - X[0]) * (Y[2] - Y[0]) - (X[2] - X[0]) * (Y[1] - Y[0]);
        if (area2 == 0)
            return;
        const int64_t sign = area2 > 0 ? 1 : -1;
        const double invArea = 1.0 / static_cast<double>(area2 * sign);

        int64_t ex[3], ey[3], ec[3];
        int64_t tlBias[3];
        for (int i = 0; i < 3; ++i)
        {
            const int a = (i + 1) % 3;
            const int b = (i + 2) % 3;
            ex[i] = -(Y[b] - Y[a]) * sign;
            ey[i] = (X[b] - X[a]) * sign;
            ec[i] = (-(X[b] - X[a]) * Y[a] + (Y[b] - Y[a]) * X[a]) * sign;
            const bool topLeft = ex[i] > 0 || (ex[i] == 0 && ey[i] > 0);
            // inside <=> w > 0 || (w == 0 && topLeft) <=> w >= tlBias
            tlBias[i] = topLeft ? 0 : 1;
        }

        const int64_t minXf = std::min({X[0], X[1], X[2]});
        const int64_t maxXf = std::max({X[0], X[1], X[2]});
        const int64_t minYf = std::min({Y[0], Y[1], Y[2]});
        const int64_t maxYf = std::max({Y[0], Y[1], Y[2]});
        const int minX = std::max<int>(ceilFixed4(static_cast<int32_t>(minXf)), ctx.scissor.x0);
        const int maxX = std::min<int>(ceilFixed4(static_cast<int32_t>(maxXf)) - 1, ctx.scissor.x1);
        const int minY = std::max<int>(ceilFixed4(static_cast<int32_t>(minYf)), ctx.scissor.y0);
        const int maxY = std::min<int>(ceilFixed4(static_cast<int32_t>(maxYf)) - 1, ctx.scissor.y1);
        if (minX > maxX || minY > maxY)
            return;

        const GSVertex &v0 = *vtx[0];
        const GSVertex &v1 = *vtx[1];
        const GSVertex &v2 = *vtx[2];
        const double texW = static_cast<double>(state.textureWidth);
        const double texH = static_cast<double>(state.textureHeight);
        const bool iip = state.prim.iip;
        const bool tme = state.prim.tme;
        const bool fst = state.prim.fst;
        const bool fge = state.prim.fge;
        const auto &tex0 = ctx.tex0;

        const int64_t stepX[3] = {ex[0] * 16, ex[1] * 16, ex[2] * 16};

#if defined(PS2X_GS_SIMD_SPAN)
        // SIMD span attributes: the edge values are carried as doubles, exact while they stay
        // integers below 2^53 (checked at the bounding-box corners; they are linear in x and y).
        bool simd = SpanSimdUsable();
        TriAttrs attrs;
        SpanAttrs span;
        if (simd)
        {
            constexpr int64_t kLimit = int64_t(1) << 50;
            for (int i = 1; i < 3 && simd; ++i)
                for (int64_t cy : {static_cast<int64_t>(minY), static_cast<int64_t>(maxY)})
                    for (int64_t cxv : {static_cast<int64_t>(minX), static_cast<int64_t>(maxX) + 4})
                    {
                        const int64_t w = stepX[i] * cxv + ey[i] * cy * 16 + ec[i];
                        if (w >= kLimit || w <= -kLimit)
                            simd = false;
                    }
        }
        if (simd)
        {
            attrs.invArea = invArea;
            attrs.texW = texW;
            attrs.texH = texH;
            attrs.iip = iip;
            attrs.tme = tme;
            attrs.fst = fst;
            attrs.fge = fge;
            attrs.Set(kChZ, v0.z, v1.z, v2.z);
            attrs.Set(kChR, v0.r, v1.r, v2.r);
            attrs.Set(kChG, v0.g, v1.g, v2.g);
            attrs.Set(kChB, v0.b, v1.b, v2.b);
            attrs.Set(kChA, v0.a, v1.a, v2.a);
            if (fst)
            {
                attrs.Set(kChU, v0.u, v1.u, v2.u);
                attrs.Set(kChV, v0.v, v1.v, v2.v);
            }
            else
            {
                attrs.Set(kChU, v0.s, v1.s, v2.s);
                attrs.Set(kChV, v0.t, v1.t, v2.t);
            }
            attrs.Set(kChQ, v0.q, v1.q, v2.q);
            attrs.Set(kChFog, v0.fog, v1.fog, v2.fog);
        }
#endif

        // Per-row span limits: for an edge with A = stepX != 0 the bound is
        // ceil((tlBias - B) / A) (A > 0) or floor((B - tlBias) / -A) (A < 0), B = ey*16*y + ec.
        // The numerator changes by a constant per row, so the exact floor quotient/remainder
        // pair is stepped incrementally instead of dividing on every row.
        struct EdgeStep
        {
            int64_t q, r, d, dq, dr;
        } es[3];
        // (Re)start the exact floor quotient/remainder of every edge at row y.
        auto startEdges = [&](int y)
        {
            for (int i = 0; i < 3; ++i)
            {
                const int64_t A = stepX[i];
                if (A == 0)
                    continue;
                const int64_t B0 = ey[i] * (static_cast<int64_t>(y) * 16) + ec[i];
                const int64_t n0 = A > 0 ? tlBias[i] - B0 : B0 - tlBias[i];
                EdgeStep &e = es[i];
                e.q = floorDiv64(n0, e.d);
                e.r = n0 - e.q * e.d;
            }
        };
        for (int i = 0; i < 3; ++i)
        {
            const int64_t A = stepX[i];
            if (A == 0)
                continue;
            const int64_t delta = A > 0 ? -ey[i] * 16 : ey[i] * 16;
            EdgeStep &e = es[i];
            e.d = A > 0 ? A : -A;
            e.dq = floorDiv64(delta, e.d);
            e.dr = delta - e.dq * e.d;
        }

        // Rows owned by this band: whole runs of consecutive rows, the edges restarted at the
        // first row of each run (the same values incremental stepping from minY would reach).
        int runStart = band.NextRow(minY);
        int runEnd = std::min(maxY, band.RunEnd(runStart));
        if (runStart <= maxY)
            startEdges(runStart);
        for (int y = runStart; y <= maxY; ++y)
        {
            if (y > runEnd)
            {
                runStart = band.NextRow(y);
                if (runStart > maxY)
                    break;
                runEnd = std::min(maxY, band.RunEnd(runStart));
                y = runStart;
                startEdges(y);
            }
            else if (y != runStart)
            {
                for (int i = 0; i < 3; ++i)
                {
                    if (stepX[i] == 0)
                        continue;
                    EdgeStep &e = es[i];
                    e.q += e.dq;
                    e.r += e.dr;
                    if (e.r >= e.d)
                    {
                        e.r -= e.d;
                        ++e.q;
                    }
                }
            }
            const int64_t py = static_cast<int64_t>(y) * 16;
            int64_t lo = minX;
            int64_t hi = maxX;
            int64_t rowBase[3];
            for (int i = 0; i < 3; ++i)
            {
                const int64_t B = ey[i] * py + ec[i];
                rowBase[i] = B;
                const int64_t A = stepX[i];
                if (A > 0)
                    lo = std::max(lo, es[i].q + (es[i].r != 0 ? 1 : 0));
                else if (A < 0)
                    hi = std::min(hi, es[i].q);
                else if (B < tlBias[i])
                    hi = lo - 1;
            }
            if (lo > hi)
                continue;
            W::PrefetchSpan(ps, static_cast<int>(lo), static_cast<int>(hi), y);
            if (track && tme)
                track->Row(y, static_cast<int>(hi));

            int64_t w1 = rowBase[1] + stepX[1] * lo;
            int64_t w2 = rowBase[2] + stepX[2] * lo;
#if defined(PS2X_GS_SIMD_SPAN)
            if (simd)
            {
                for (int64_t cx = lo; cx <= hi; cx += kSpanChunk)
                {
                    const int n = static_cast<int>(std::min<int64_t>(kSpanChunk, hi - cx + 1));
                    const int64_t cw1 = rowBase[1] + stepX[1] * cx;
                    const int64_t cw2 = rowBase[2] + stepX[2] * cx;
                    ComputeSpanAttrs(attrs, static_cast<double>(cw1), static_cast<double>(cw2),
                                     static_cast<double>(stepX[1]), static_cast<double>(stepX[2]), n, span);
                    for (int i = 0; i < n; ++i)
                    {
                        const int x = static_cast<int>(cx) + i;
                        uint8_t r, g, b, a;
                        if (iip)
                        {
                            r = static_cast<uint8_t>(span.r[i]);
                            g = static_cast<uint8_t>(span.g[i]);
                            b = static_cast<uint8_t>(span.b[i]);
                            a = static_cast<uint8_t>(span.a[i]);
                        }
                        else
                        {
                            r = v2.r;
                            g = v2.g;
                            b = v2.b;
                            a = v2.a;
                        }
                        if (tme)
                        {
                            const uint32_t texel = sample(smp, span.uf[i], span.vf[i], fst ? 1.0f : span.q[i]);
                            const TextureCombineResult color = fastCombine(tex0, r, g, b, a, texel);
                            r = color.r;
                            g = color.g;
                            b = color.b;
                            a = color.a;
                        }
                        const uint8_t fog = fge ? static_cast<uint8_t>(span.fog[i]) : 0u;
                        W::Write(ps, x, y, span.z[i], r, g, b, a, fog);
                    }
                }
                continue;
            }
#endif
            for (int x = static_cast<int>(lo); x <= static_cast<int>(hi); ++x, w1 += stepX[1], w2 += stepX[2])
            {

                const double b1 = static_cast<double>(w1) * invArea;
                const double b2 = static_cast<double>(w2) * invArea;

                const uint32_t z = vertexZ(std::floor(interpolate3(v0.z, v1.z, v2.z, b1, b2) + 1.0e-6));

                uint8_t r, g, b, a;
                if (iip)
                {
                    r = clampU8(truncateAttribute(interpolate3(v0.r, v1.r, v2.r, b1, b2)));
                    g = clampU8(truncateAttribute(interpolate3(v0.g, v1.g, v2.g, b1, b2)));
                    b = clampU8(truncateAttribute(interpolate3(v0.b, v1.b, v2.b, b1, b2)));
                    a = clampU8(truncateAttribute(interpolate3(v0.a, v1.a, v2.a, b1, b2)));
                }
                else
                {
                    r = v2.r;
                    g = v2.g;
                    b = v2.b;
                    a = v2.a;
                }

                if (tme)
                {
                    int32_t uf, vf;
                    float q = 1.0f;
                    if (fst)
                    {
                        uf = textureCoordToFixed(interpolate3(v0.u, v1.u, v2.u, b1, b2) / 16.0);
                        vf = textureCoordToFixed(interpolate3(v0.v, v1.v, v2.v, b1, b2) / 16.0);
                    }
                    else
                    {
                        const double is = interpolate3(v0.s, v1.s, v2.s, b1, b2);
                        const double it = interpolate3(v0.t, v1.t, v2.t, b1, b2);
                        double iq = interpolate3(v0.q, v1.q, v2.q, b1, b2);
                        if (iq == 0.0)
                            iq = 1.0e-30;
                        uf = textureCoordToFixed(is / iq * texW);
                        vf = textureCoordToFixed(it / iq * texH);
                        q = static_cast<float>(iq);
                    }

                    const uint32_t texel = sample(smp, uf, vf, q);
                    const TextureCombineResult color = fastCombine(tex0, r, g, b, a, texel);
                    r = color.r;
                    g = color.g;
                    b = color.b;
                    a = color.a;
                }

                const uint8_t fog = fge ? clampU8(truncateAttribute(interpolate3(v0.fog, v1.fog, v2.fog, b1, b2))) : 0u;
                W::Write(ps, x, y, z, r, g, b, a, fog);
            }
        }
    }

    template <class W>
    static void RasterSprite(const GSPrimitiveBatch &batch, const FastPixelState &psIn, const FastSampler &smp, FastSampleFn sample, const FastBand &band, FastTrack *track)
    {
        // Local copy: the per-pixel sampler call cannot modify it, so state loads and the row-
        // invariant address math can stay hoisted out of the pixel loop.
        const FastPixelState ps = psIn;
        const GSDrawState &state = batch.state;
        const auto &ctx = state.context;
        const GSVertex &first = batch.vertices[0];
        const GSVertex &second = batch.vertices[1];

        int32_t px0 = vertexFixedX(first, ctx);
        int32_t py0 = vertexFixedY(first, ctx);
        int32_t px1 = vertexFixedX(second, ctx);
        int32_t py1 = vertexFixedY(second, ctx);

        const double texW = static_cast<double>(state.textureWidth);
        const double texH = static_cast<double>(state.textureHeight);
        const double q1 = second.q != 0.0f ? static_cast<double>(second.q) : 1.0;
        double tu0, tv0, tu1, tv1;
        if (state.prim.fst)
        {
            tu0 = first.u / 16.0;
            tv0 = first.v / 16.0;
            tu1 = second.u / 16.0;
            tv1 = second.v / 16.0;
        }
        else
        {
            tu0 = first.s / q1 * texW;
            tv0 = first.t / q1 * texH;
            tu1 = second.s / q1 * texW;
            tv1 = second.t / q1 * texH;
        }

        if (px0 > px1)
        {
            std::swap(px0, px1);
            std::swap(tu0, tu1);
        }
        if (py0 > py1)
        {
            std::swap(py0, py1);
            std::swap(tv0, tv1);
        }

        int left = ceilFixed4(px0);
        int right = ceilFixed4(px1);
        int top = ceilFixed4(py0);
        int bottom = ceilFixed4(py1);
        left = std::max<int>(left, ctx.scissor.x0);
        top = std::max<int>(top, ctx.scissor.y0);
        right = std::min<int>(right, static_cast<int>(ctx.scissor.x1) + 1);
        bottom = std::min<int>(bottom, static_cast<int>(ctx.scissor.y1) + 1);
        if (left >= right || top >= bottom)
            return;

        const uint32_t z = vertexZ(second.z);
        const uint8_t r = second.r, g = second.g, b = second.b, a = second.a;

        if (!state.prim.tme)
        {
            for (int y = top; y < bottom; ++y)
            {
                if (!band.Has(y))
                    continue;
                for (int x = left; x < right; ++x)
                    W::Write(ps, x, y, z, r, g, b, a, second.fog);
            }
            return;
        }

        const double dudx = (px1 != px0) ? (tu1 - tu0) * 16.0 / static_cast<double>(px1 - px0) : 0.0;
        const double dvdy = (py1 != py0) ? (tv1 - tv0) * 16.0 / static_cast<double>(py1 - py0) : 0.0;
        const double ox = static_cast<double>(px0) / 16.0;
        const double oy = static_cast<double>(py0) / 16.0;
        const float q = static_cast<float>(q1);
        const auto &tex0 = ctx.tex0;

        // The U coordinate depends on x only: compute it once per column (same expression).
        constexpr int kMaxCols = 2048;
        int32_t ufCol[kMaxCols];
        const bool hoistU = right - left <= kMaxCols;
        bool ufReady = false;
        for (int y = top; y < bottom; ++y)
        {
            if (!band.Has(y))
                continue;
            if (track)
                track->Row(y, right - 1);
            const int32_t vf = textureCoordToFixed(tv0 + dvdy * (static_cast<double>(y) - oy));
            if (hoistU)
            {
                if (!ufReady)
                {
                    for (int x = left; x < right; ++x)
                        ufCol[x - left] = textureCoordToFixed(tu0 + dudx * (static_cast<double>(x) - ox));
                    ufReady = true;
                }
                for (int x = left; x < right; ++x)
                {
                    const uint32_t texel = sample(smp, ufCol[x - left], vf, q);
                    const TextureCombineResult color = fastCombine(tex0, r, g, b, a, texel);
                    W::Write(ps, x, y, z, color.r, color.g, color.b, color.a, second.fog);
                }
                continue;
            }
            for (int x = left; x < right; ++x)
            {
                const int32_t uf = textureCoordToFixed(tu0 + dudx * (static_cast<double>(x) - ox));
                const uint32_t texel = sample(smp, uf, vf, q);
                const TextureCombineResult color = fastCombine(tex0, r, g, b, a, texel);
                W::Write(ps, x, y, z, color.r, color.g, color.b, color.a, second.fog);
            }
        }
    }

    using RasterFn = void (*)(const GSPrimitiveBatch &, const FastPixelState &, const FastSampler &, FastSampleFn, const FastBand &, FastTrack *);

    template <bool SPRITE, uint8_t FPSM, uint8_t ZPSM>
    static void RasterWith(const GSPrimitiveBatch &batch, const FastPixelState &ps, const FastSampler &smp, FastSampleFn sample,
                           const FastBand &band, FastTrack *track)
    {
        if constexpr (SPRITE)
            RasterSprite<FastWriter<FPSM, ZPSM>>(batch, ps, smp, sample, band, track);
        else
            RasterTriangle<FastWriter<FPSM, ZPSM>>(batch, ps, smp, sample, band, track);
    }

    template <bool SPRITE, uint8_t FPSM>
    static RasterFn SelectZ(uint8_t zKind)
    {
        switch (zKind)
        {
        case GS_PSM_Z32: return &RasterWith<SPRITE, FPSM, GS_PSM_Z32>;
        case GS_PSM_Z24: return &RasterWith<SPRITE, FPSM, GS_PSM_Z24>;
        case GS_PSM_Z16: return &RasterWith<SPRITE, FPSM, GS_PSM_Z16>;
        case GS_PSM_Z16S: return &RasterWith<SPRITE, FPSM, GS_PSM_Z16S>;
        default: return &RasterWith<SPRITE, FPSM, kZNone>;
        }
    }

    template <bool SPRITE>
    static RasterFn SelectRaster(uint8_t fpsm, uint8_t zKind)
    {
        switch (fpsm)
        {
        case GS_PSM_CT32: return SelectZ<SPRITE, GS_PSM_CT32>(zKind);
        case GS_PSM_CT24: return SelectZ<SPRITE, GS_PSM_CT24>(zKind);
        case GS_PSM_CT16: return SelectZ<SPRITE, GS_PSM_CT16>(zKind);
        case GS_PSM_CT16S: return SelectZ<SPRITE, GS_PSM_CT16S>(zKind);
        default: return nullptr;
        }
    }

    // Clipped pixel bounds exactly as the rasterizers compute them; false when nothing is drawn.
    static bool TriangleBounds(const GSPrimitiveBatch &batch, int &x0, int &x1, int &y0, int &y1)
    {
        const auto &ctx = batch.state.context;
        int64_t X[3], Y[3];
        for (int i = 0; i < 3; ++i)
        {
            X[i] = vertexFixedX(batch.vertices[i], ctx);
            Y[i] = vertexFixedY(batch.vertices[i], ctx);
        }
        const int64_t area2 = (X[1] - X[0]) * (Y[2] - Y[0]) - (X[2] - X[0]) * (Y[1] - Y[0]);
        if (area2 == 0)
            return false;
        const int64_t minXf = std::min({X[0], X[1], X[2]});
        const int64_t maxXf = std::max({X[0], X[1], X[2]});
        const int64_t minYf = std::min({Y[0], Y[1], Y[2]});
        const int64_t maxYf = std::max({Y[0], Y[1], Y[2]});
        x0 = std::max<int>(ceilFixed4(static_cast<int32_t>(minXf)), ctx.scissor.x0);
        x1 = std::min<int>(ceilFixed4(static_cast<int32_t>(maxXf)) - 1, ctx.scissor.x1);
        y0 = std::max<int>(ceilFixed4(static_cast<int32_t>(minYf)), ctx.scissor.y0);
        y1 = std::min<int>(ceilFixed4(static_cast<int32_t>(maxYf)) - 1, ctx.scissor.y1);
        return x0 <= x1 && y0 <= y1;
    }

    static bool SpriteBounds(const GSPrimitiveBatch &batch, int &x0, int &x1, int &y0, int &y1)
    {
        const auto &ctx = batch.state.context;
        int32_t px0 = vertexFixedX(batch.vertices[0], ctx);
        int32_t py0 = vertexFixedY(batch.vertices[0], ctx);
        int32_t px1 = vertexFixedX(batch.vertices[1], ctx);
        int32_t py1 = vertexFixedY(batch.vertices[1], ctx);
        if (px0 > px1)
            std::swap(px0, px1);
        if (py0 > py1)
            std::swap(py0, py1);
        x0 = std::max<int>(ceilFixed4(px0), ctx.scissor.x0);
        y0 = std::max<int>(ceilFixed4(py0), ctx.scissor.y0);
        x1 = std::min<int>(ceilFixed4(px1), static_cast<int>(ctx.scissor.x1) + 1) - 1;
        y1 = std::min<int>(ceilFixed4(py1), static_cast<int>(ctx.scissor.y1) + 1) - 1;
        return x0 <= x1 && y0 <= y1;
    }

    // Set of 8 KiB VRAM pages.
    struct PageSet
    {
        uint64_t bits[GSMem::MEMORY_SIZE / GSMem::TexturePageCache::kPageSize / 64u]{};
        static constexpr uint32_t kPages = GSMem::MEMORY_SIZE / GSMem::TexturePageCache::kPageSize;

        void Set(uint32_t page)
        {
            page %= kPages;
            bits[page >> 6] |= 1ull << (page & 63u);
        }
        bool Test(uint32_t page) const
        {
            page %= kPages;
            return (bits[page >> 6] >> (page & 63u)) & 1u;
        }
        bool Intersects(const PageSet &o) const
        {
            uint64_t acc = 0;
            for (size_t i = 0; i < std::size(bits); ++i)
                acc |= bits[i] & o.bits[i];
            return acc != 0;
        }
        void Merge(const PageSet &o)
        {
            for (size_t i = 0; i < std::size(bits); ++i)
                bits[i] |= o.bits[i];
        }
        void Clear() { std::memset(bits, 0, sizeof(bits)); }
    };

    // Pages holding the pixels of [x0,x1]x[y0,y1] in a buffer. With exclusive=true it fails when
    // two different pixels of the rectangle could share an address (row overflow or 4 MiB wrap).
    template <uint8_t PSM>
    static bool AddRectPages(PageSet &set, uint32_t bp, uint32_t bw, int x0, int x1, int y0, int y1, bool exclusive)
    {
        using Traits = typename FastPsm<PSM>::Traits;
        constexpr uint32_t pw = Traits::PageExtent().x;
        constexpr uint32_t ph = Traits::PageExtent().y;
        const uint32_t basePage = bp / 32u;
        const uint32_t rowPages = (bw * 64u) / pw;
        const uint32_t c0 = static_cast<uint32_t>(x0) / pw, c1 = static_cast<uint32_t>(x1) / pw;
        const uint32_t r0 = static_cast<uint32_t>(y0) / ph, r1 = static_cast<uint32_t>(y1) / ph;
        if (exclusive)
        {
            if (c1 >= rowPages)
                return false;
            if ((r1 * rowPages + c1) - (r0 * rowPages + c0) >= PageSet::kPages)
                return false;
        }
        const bool spill = (bp % 32u) != 0u; // block offset can carry into the next page
        uint32_t count = 0;
        for (uint32_t r = r0; r <= r1; ++r)
            for (uint32_t c = c0; c <= c1; ++c)
            {
                const uint32_t page = basePage + r * rowPages + c;
                set.Set(page);
                if (spill)
                    set.Set(page + 1u);
                if (++count > 2u * PageSet::kPages)
                    return !exclusive; // covers everything already
            }
        return true;
    }

    static bool AddTexRectPages(PageSet &set, uint8_t psm, uint32_t bp, uint32_t bw, int x0, int x1, int y0, int y1)
    {
        switch (psm)
        {
        case GS_PSM_CT32: return AddRectPages<GS_PSM_CT32>(set, bp, bw, x0, x1, y0, y1, false);
        case GS_PSM_CT24: return AddRectPages<GS_PSM_CT24>(set, bp, bw, x0, x1, y0, y1, false);
        case GS_PSM_CT16: return AddRectPages<GS_PSM_CT16>(set, bp, bw, x0, x1, y0, y1, false);
        case GS_PSM_CT16S: return AddRectPages<GS_PSM_CT16S>(set, bp, bw, x0, x1, y0, y1, false);
        case GS_PSM_T8: return AddRectPages<GS_PSM_T8>(set, bp, bw, x0, x1, y0, y1, false);
        case GS_PSM_T4: return AddRectPages<GS_PSM_T4>(set, bp, bw, x0, x1, y0, y1, false);
        case GS_PSM_T8H: return AddRectPages<GS_PSM_T8H>(set, bp, bw, x0, x1, y0, y1, false);
        case GS_PSM_T4HL: return AddRectPages<GS_PSM_T4HL>(set, bp, bw, x0, x1, y0, y1, false);
        case GS_PSM_T4HH: return AddRectPages<GS_PSM_T4HH>(set, bp, bw, x0, x1, y0, y1, false);
        case GS_PSM_Z32: return AddRectPages<GS_PSM_Z32>(set, bp, bw, x0, x1, y0, y1, false);
        case GS_PSM_Z24: return AddRectPages<GS_PSM_Z24>(set, bp, bw, x0, x1, y0, y1, false);
        case GS_PSM_Z16: return AddRectPages<GS_PSM_Z16>(set, bp, bw, x0, x1, y0, y1, false);
        case GS_PSM_Z16S: return AddRectPages<GS_PSM_Z16S>(set, bp, bw, x0, x1, y0, y1, false);
        default: return false;
        }
    }

    template <uint8_t FPSM>
    static bool AddFramePages(PageSet &set, uint32_t bp, uint32_t bw, int x0, int x1, int y0, int y1)
    {
        return AddRectPages<FPSM>(set, bp, bw, x0, x1, y0, y1, true);
    }

    static bool AddTargetPages(PageSet &set, uint8_t psm, uint32_t bp, uint32_t bw, int x0, int x1, int y0, int y1)
    {
        switch (psm)
        {
        case GS_PSM_CT32: return AddFramePages<GS_PSM_CT32>(set, bp, bw, x0, x1, y0, y1);
        case GS_PSM_CT24: return AddFramePages<GS_PSM_CT24>(set, bp, bw, x0, x1, y0, y1);
        case GS_PSM_CT16: return AddFramePages<GS_PSM_CT16>(set, bp, bw, x0, x1, y0, y1);
        case GS_PSM_CT16S: return AddFramePages<GS_PSM_CT16S>(set, bp, bw, x0, x1, y0, y1);
        case GS_PSM_Z32: return AddFramePages<GS_PSM_Z32>(set, bp, bw, x0, x1, y0, y1);
        case GS_PSM_Z24: return AddFramePages<GS_PSM_Z24>(set, bp, bw, x0, x1, y0, y1);
        case GS_PSM_Z16: return AddFramePages<GS_PSM_Z16>(set, bp, bw, x0, x1, y0, y1);
        case GS_PSM_Z16S: return AddFramePages<GS_PSM_Z16S>(set, bp, bw, x0, x1, y0, y1);
        default: return false;
        }
    }

    static void LevelRange(uint8_t wrap, int size, uint16_t mn, uint16_t mx, int &lo, int &hi)
    {
        switch (wrap & 3u)
        {
        case 0:
        case 1:
            lo = 0;
            hi = size - 1;
            break;
        case 2:
            lo = std::min<int>(mn, mx);
            hi = mx;
            break;
        default:
            lo = mx;
            hi = mn | mx;
            break;
        }
    }

    // Every page the sampler can read (all levels it may select).
    static bool TexturePages(const GSDrawState &state, const FastSampler &smp, PageSet &set)
    {
        const auto &ctx = state.context;
        const uint32_t mxl = static_cast<uint32_t>((ctx.tex1 >> 2) & 0x7u);
        const uint32_t levels = mxl == 0u ? 1u : std::min<uint32_t>(mxl, 6u) + 1u;
        for (uint32_t level = 0; level < levels; ++level)
        {
            const FastLevel &lv = smp.lv[level];
            int u0, u1, v0, v1;
            LevelRange(smp.wrapU, lv.texW, lv.minU, lv.maxU, u0, u1);
            LevelRange(smp.wrapV, lv.texH, lv.minV, lv.maxV, v0, v1);
            if (!AddTexRectPages(set, ctx.tex0.psm, lv.tbp, lv.tbw, u0, u1, v0, v1))
                return false;
        }
        return true;
    }

    template <bool SPRITE>
    static bool Draw(GSCpuBackend &be, const GSPrimitiveBatch &batch);
};

// Parallel rasterization: eligible primitives are collected into a batch and drawn by several
// threads. The rows are split into 2^bandShift-row bands, interleaved into `groups` row groups
// (band b belongs to group b % groups); threads take whole groups from a shared counter and draw
// every primitive of the batch, in submission order, restricted to the group's rows. Each pixel is
// therefore written by one thread in serial primitive order, whichever thread takes its group.
// A primitive is eligible when the batch provably behaves like serial execution:
//  - every primitive of the batch targets the same frame/depth buffers, the frame and depth
//    pages are disjoint and no two pixels of a primitive share an address;
//  - no page that may be read as texture is written by the batch (texture reads then see the
//    batch-start VRAM, as serially) and texture sampling does not need the shared page cache;
// The texture page cache is updated to the state the serial run leaves (the page of the last
// texel fetched in raster order), and a pending copy is taken before the batch when the batch
// writes the cached page.
struct GSCpuBackend::ParallelState
{
    using PageSet = GSFastRaster::PageSet;
    using RasterFn = GSFastRaster::RasterFn;
    struct Job
    {
        GSPrimitiveBatch batch;
        FastPixelState ps;
        FastSampler smp;
        FastSampleFn sample = nullptr;
        RasterFn raster = nullptr;
        int y0 = 0, y1 = 0;
        bool tme = false;
    };

    struct ThreadResult
    {
        bool any = false;
        size_t job = 0;
        int y = 0, x = 0;
        uint32_t page = 0;
    };

    static constexpr size_t kMaxJobs = 512;

    GSCpuBackend &be;
    std::vector<Job> jobs;
    size_t count = 0;
    std::vector<std::array<uint32_t, 256>> palettes;
    size_t paletteCount = 0;
    const uint32_t *lastPaletteSrc = nullptr;
    uint64_t lastPaletteKey = ~0ull, lastPaletteVersion = ~0ull;

    PageSet fbPages, zPages, texPages;
    uint64_t configKey = 0;

    // Texture page cache operations (CLUT load, TEXFLUSH) executed while jobs were pending:
    // the first job queued after the last one. The main cache already holds the state that
    // operation left; it only changes again if a later job fetches a texel. kNoMarker: none.
    static constexpr size_t kNoMarker = SIZE_MAX;
    size_t cacheMarker = kNoMarker;

    // Cached texture page set of the last textured primitive.
    struct TexKey
    {
        uint64_t a = ~0ull, b = 0, c = 0, d = 0, e = 0;
        bool operator==(const TexKey &o) const { return a == o.a && b == o.b && c == o.c && d == o.d && e == o.e; }
    } lastTexKey;
    PageSet lastTexPages;
    bool lastTexOk = false;

    uint32_t threads = 1;
    uint32_t bandShift = 3;
    uint32_t groups = 1;
    std::atomic<uint32_t> nextGroup{0};
    std::vector<std::thread> helpers;
    std::vector<std::unique_ptr<GSMem::TexturePageCache>> caches;
    std::vector<ThreadResult> results;
    std::atomic<uint64_t> generation{0};
    std::atomic<uint32_t> remaining{0};
    std::mutex wakeMutex;
    std::condition_variable wakeCv;
    std::atomic<bool> stop{false};

    explicit ParallelState(GSCpuBackend &backend, uint32_t threadCount, uint32_t shift)
        : be(backend), threads(threadCount), bandShift(shift)
    {
        static const uint32_t groupFactor = []()
        {
            const char *e = std::getenv("PS2X_GS_GROUP_FACTOR");
            return e ? static_cast<uint32_t>(std::clamp(std::atoi(e), 1, 64)) : 2u;
        }();
        groups = threads * groupFactor;
        jobs.resize(kMaxJobs);
        palettes.resize(kMaxJobs);
        results.resize(groups);
        for (uint32_t i = 0; i < threads; ++i)
            caches.push_back(std::make_unique<GSMem::TexturePageCache>());
        for (uint32_t i = 1; i < threads; ++i)
            helpers.emplace_back([this, i]()
                                 { HelperMain(i); });
    }

    ~ParallelState()
    {
        {
            std::lock_guard<std::mutex> lk(wakeMutex);
            stop.store(true);
        }
        wakeCv.notify_all();
        for (auto &t : helpers)
            t.join();
    }

    void HelperMain(uint32_t index)
    {
        uint64_t seen = 0;
        for (;;)
        {
            uint64_t g = generation.load(std::memory_order_acquire);
            for (int spins = 0; g == seen && spins < 4000; ++spins)
            {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#endif
                g = generation.load(std::memory_order_acquire);
            }
            if (g == seen)
            {
                std::unique_lock<std::mutex> lk(wakeMutex);
                wakeCv.wait(lk, [&]()
                            { return stop.load() || generation.load(std::memory_order_acquire) != seen; });
                if (stop.load())
                    return;
                g = generation.load(std::memory_order_acquire);
            }
            seen = g;
            RunGroups(index);
            remaining.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    // Draws every row group not yet taken by another thread.
    void RunGroups(uint32_t index)
    {
        for (;;)
        {
            const uint32_t g = nextGroup.fetch_add(1, std::memory_order_relaxed);
            if (g >= groups)
                return;
            RunGroup(g, index);
        }
    }

    void RunGroup(uint32_t group, uint32_t index)
    {
        FastBand band{bandShift, groups, group};
        GSMem::TexturePageCache &tc = *caches[index];
        tc.Invalidate();
        ThreadResult res;
        const uint32_t period = (1u << bandShift) * groups;
        for (size_t j = 0; j < count; ++j)
        {
            const Job &job = jobs[j];
            if (static_cast<uint32_t>(job.y1 - job.y0) + 1u < period && !band.Overlaps(job.y0, job.y1))
                continue;
            if (job.tme)
            {
                FastSampler s = job.smp;
                s.cache = &tc;
                FastTrack track;
                job.raster(job.batch, job.ps, s, job.sample, band, &track);
                if (track.any)
                {
                    res.any = true;
                    res.job = j;
                    res.y = track.y;
                    res.x = track.x;
                    res.page = tc.PageBase();
                }
            }
            else
                job.raster(job.batch, job.ps, job.smp, job.sample, band, nullptr);
        }
        results[group] = res;
    }

    const uint32_t *StorePalette(const uint32_t *src, uint64_t key, uint64_t version)
    {
        if (paletteCount != 0u && src == lastPaletteSrc && key == lastPaletteKey && version == lastPaletteVersion)
            return palettes[paletteCount - 1u].data();
        std::memcpy(palettes[paletteCount].data(), src, 256u * sizeof(uint32_t));
        lastPaletteSrc = src;
        lastPaletteKey = key;
        lastPaletteVersion = version;
        return palettes[paletteCount++].data();
    }

    void Run()
    {
        if (count == 0u)
            return;
        GSMem::TexturePageCache &cache = be.m_texturePageCache;
        // With a cache marker, p0 is the page left by the marked operation (TEXFLUSH: none; CLUT
        // load: its last page, never written by the batch and not materialized), so the
        // adjustments below are no-ops and the batch always runs in parallel.
        cache.Unmaterialize(be.m_vram);
        const uint32_t p0 = cache.PageBase();
        bool parallel = true;
        if (p0 != GSMem::TexturePageCache::kNoPage)
        {
            const uint32_t page0 = p0 / GSMem::TexturePageCache::kPageSize;
            // The serial run would copy the cached page before the batch's first write to it.
            if (!cache.Materialized() && (fbPages.Test(page0) || zPages.Test(page0)))
                cache.PrepareUnknownWrite(be.m_vram);
            // A stale copy that the batch reads cannot be reproduced in parallel.
            if (cache.Materialized() && texPages.Test(page0))
                parallel = false;
        }

        if (!parallel || threads <= 1u)
        {
            FastBand all;
            for (size_t j = 0; j < count; ++j)
            {
                const Job &job = jobs[j];
                job.raster(job.batch, job.ps, job.smp, job.sample, all, nullptr);
            }
        }
        else
        {
            nextGroup.store(0u, std::memory_order_relaxed);
            remaining.store(threads - 1u, std::memory_order_release);
            generation.fetch_add(1, std::memory_order_acq_rel);
            {
                std::lock_guard<std::mutex> lk(wakeMutex);
            }
            wakeCv.notify_all();
            RunGroups(0);
            while (remaining.load(std::memory_order_acquire) != 0u)
            {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#endif
            }
            const ThreadResult *best = nullptr;
            for (const ThreadResult &r : results)
            {
                // r is the thread's last textured job; one before the marker leaves the cache
                // as the marked operation set it.
                if (!r.any || (cacheMarker != kNoMarker && r.job < cacheMarker))
                    continue;
                if (!best || r.job > best->job || (r.job == best->job && (r.y > best->y || (r.y == best->y && r.x > best->x))))
                    best = &r;
            }
            if (best && !GSMem::TexturePageCache::Disabled())
                cache.SetPending(best->page);
        }
        count = 0;
        cacheMarker = kNoMarker;
        paletteCount = 0;
        lastPaletteSrc = nullptr;
        fbPages.Clear();
        zPages.Clear();
        texPages.Clear();
    }
};

template <bool SPRITE>
bool GSFastRaster::Draw(GSCpuBackend &be, const GSPrimitiveBatch &batch)
{
    FastPixelState ps;
    uint8_t zKind = kZNone;
    if (!SetupPixelState(be, batch.state, ps, zKind))
        return false;
    FastSampler smp;
    FastSampleFn sample = nullptr;
    const bool tme = batch.state.prim.tme;
    if (tme)
        sample = SetupSampler(be, batch.state, smp);
    const uint8_t fpsm = batch.state.context.frame.psm;
    const RasterFn raster = SelectRaster<SPRITE>(fpsm, zKind);
    if (!raster)
        return false;

    GSCpuBackend::ParallelState *par = be.ParallelFor();
    if (par && !(tme && sample == &SampleGeneric))
    {
        int x0, x1, y0, y1;
        const bool drawn = SPRITE ? SpriteBounds(batch, x0, x1, y0, y1) : TriangleBounds(batch, x0, x1, y0, y1);
        if (!drawn)
            return true; // covers no pixel: no texel fetch, no write

        PageSet fb, zb, tex;
        bool ok = AddTargetPages(fb, fpsm, ps.fbp, ps.fbw, x0, x1, y0, y1);
        if (ok && zKind != kZNone)
            ok = AddTargetPages(zb, zKind, ps.zbp, ps.fbw, x0, x1, y0, y1);
        if (ok && tme)
        {
            const auto &ctx = batch.state.context;
            GSCpuBackend::ParallelState::TexKey key;
            key.a = (static_cast<uint64_t>(ctx.tex0.tbp0) << 32) | (static_cast<uint64_t>(ctx.tex0.tbw) << 24) |
                    (static_cast<uint64_t>(ctx.tex0.psm) << 16) | (static_cast<uint64_t>(batch.state.textureWidth) << 0);
            key.b = ctx.clamp;
            key.c = ctx.miptbp1;
            key.d = ctx.miptbp2;
            key.e = (static_cast<uint64_t>(batch.state.textureHeight) << 8) | ((ctx.tex1 >> 2) & 0x7u);
            if (!(key == par->lastTexKey))
            {
                par->lastTexKey = key;
                par->lastTexPages.Clear();
                par->lastTexOk = TexturePages(batch.state, smp, par->lastTexPages);
            }
            ok = par->lastTexOk;
            tex = par->lastTexPages;
        }
        if (ok)
            ok = !fb.Intersects(zb) && !tex.Intersects(fb) && !tex.Intersects(zb);
        if (ok)
        {
            const uint64_t configKey = (static_cast<uint64_t>(ps.fbp) << 40) | (static_cast<uint64_t>(ps.fbw) << 32) |
                                       (static_cast<uint64_t>(fpsm) << 24) | (static_cast<uint64_t>(ps.zbp) << 8) | zKind;
            bool conflict = par->count != 0u &&
                            (configKey != par->configKey || par->count >= GSCpuBackend::ParallelState::kMaxJobs ||
                             fb.Intersects(par->zPages) || zb.Intersects(par->fbPages) ||
                             tex.Intersects(par->fbPages) || tex.Intersects(par->zPages) ||
                             par->texPages.Intersects(fb) || par->texPages.Intersects(zb));
            if (conflict)
                par->Run();
            par->configKey = configKey;
            par->fbPages.Merge(fb);
            par->zPages.Merge(zb);
            par->texPages.Merge(tex);
            GSCpuBackend::ParallelState::Job &job = par->jobs[par->count++];
            job.batch = batch;
            job.ps = ps;
            job.ps.state = &job.batch.state;
            job.smp = smp;
            job.smp.state = &job.batch.state;
            if (tme && smp.palette)
                job.smp.palette = par->StorePalette(smp.palette, be.m_paletteKey, be.m_paletteVersion);
            job.sample = sample;
            job.raster = raster;
            job.y0 = y0;
            job.y1 = y1;
            job.tme = tme;
            return true;
        }
    }

    be.FlushParallel();
    raster(batch, ps, smp, sample, FastBand{}, nullptr);
    return true;
}

GSCpuBackend::ParallelState *GSCpuBackend::ParallelFor()
{
    if (m_parDisabled)
        return nullptr;
    if (!m_par)
    {
        static const uint32_t threads = []()
        {
            const char *e = std::getenv("PS2X_GS_THREADS");
            if (e)
                return static_cast<uint32_t>(std::max(1, std::atoi(e)));
            const uint32_t hw = std::max(1u, std::thread::hardware_concurrency());
            return std::min<uint32_t>(4u, std::max<uint32_t>(1u, hw / 4u));
        }();
        static const uint32_t shift = []()
        {
            const char *e = std::getenv("PS2X_GS_BAND_SHIFT");
            return e ? static_cast<uint32_t>(std::clamp(std::atoi(e), 0, 8)) : 3u;
        }();
        if (threads <= 1u)
        {
            m_parDisabled = true;
            return nullptr;
        }
        m_par = std::make_unique<ParallelState>(*this, threads, shift);
    }
    return m_par.get();
}

void GSCpuBackend::FlushParallel()
{
    if (m_par)
        m_par->Run();
}

// TEXFLUSH with jobs pending: in serial order the jobs before it cannot observe it, and the
// cache state it leaves (empty) is kept unless a later job fetches a texel. The pending jobs
// must not depend on the cache state the flush discards: Run() draws a batch serially when
// the cache holds a materialized (stale) copy of a page the batch reads, which it could no
// longer detect after the flush, so the batch is drawn first in that case.
void GSCpuBackend::DeferTextureFlush()
{
    ParallelState *par = m_par.get();
    if (!par || par->count == 0u)
        return;
    GSMem::TexturePageCache &cache = m_texturePageCache;
    cache.Unmaterialize(m_vram);
    if (cache.Materialized() && par->texPages.Test(cache.PageBase() / GSMem::TexturePageCache::kPageSize))
    {
        FlushParallel();
        return;
    }
    par->cacheMarker = par->count;
}

// CLUT load with jobs pending. It may run before the pending jobs when
//  - none of the pages it reads is written by them (it then reads the same VRAM bytes), and
//    the pages are protected from later jobs of the batch (merged into texPages);
//  - the texture page cache holds no materialized copy: serially the cache would then hold
//    either a pending page (reads see VRAM) or a copy taken during the batch prefix, which
//    equals the batch-start VRAM it reads now.
// Afterwards the cache is pending on the last CLUT page read, exactly as serially (that page
// is never written by the batch), until a later job fetches a texel.
// Returns false when the batch must be drawn first.
bool GSCpuBackend::DeferClutLoad(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    ParallelState *par = m_par.get();
    if (!par || par->count == 0u)
        return true;
    const bool sixteenBit = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
    const bool thirtyTwoBit = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
    if (!sixteenBit && !thirtyTwoBit)
        return true; // LoadClutUnlocked reads nothing
    m_texturePageCache.Unmaterialize(m_vram);
    if (m_texturePageCache.Materialized())
        return false;

    // Pixel rectangle LoadClutUnlocked reads (see its loop).
    GSFastRaster::PageSet pages;
    bool ok;
    if (tex0.csm == 0u)
        ok = GSFastRaster::AddTexRectPages(pages, tex0.cpsm, tex0.cbp, 1u, 0, 15, 0, 15);
    else
    {
        const uint32_t entryCount = isFourBitIndexedPsm(tex0.psm) ? 16u : 256u;
        const uint32_t width = texclut.cbw != 0u ? static_cast<uint32_t>(texclut.cbw) : 1u;
        const int x0 = static_cast<int>(static_cast<uint32_t>(texclut.cou) << 4u);
        const int y = static_cast<int>(texclut.cov);
        ok = GSFastRaster::AddTexRectPages(pages, tex0.cpsm, tex0.cbp, width, x0, x0 + static_cast<int>(entryCount) - 1, y, y);
    }
    if (!ok || pages.Intersects(par->fbPages) || pages.Intersects(par->zPages))
        return false;
    par->texPages.Merge(pages);
    par->cacheMarker = par->count;
    return true;
}

// Host->local upload with jobs pending. It may run before the pending jobs when the pages of the
// whole transfer rectangle are disjoint from every page the batch writes (frame, depth) or reads
// (textures): the VRAM results are then the same in either order. Its effect on the texture page
// cache (PrepareUnknownWrite before each write) only materializes the current page, which is
// observationally neutral: a copy and a pending page read the same bytes until the page is
// written, and the materialization any later write would trigger copies those same bytes (no
// write to that page happens in between: the upload and the batch do not write batch texture
// pages). Run() then leaves the cache on the last texel the batch fetches, exactly as serially.
// Returns false when the batch must be drawn first.
bool GSCpuBackend::DeferUpload() const
{
    const ParallelState *par = m_par.get();
    if (!par || par->count == 0u)
        return true;
    if (m_transferState.direction != 0u || !m_vram)
        return true; // the upload writes nothing
    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    if (rrw == 0u || rrh == 0u)
        return true;
    if (static_cast<uint64_t>(rrw) * rrh > 256u * 256u)
        return false;
    const int x0 = static_cast<int>(m_transfer.trxpos.dsax);
    const int y0 = static_cast<int>(m_transfer.trxpos.dsay);
    GSFastRaster::PageSet pages;
    if (!GSFastRaster::AddTexRectPages(pages, m_transfer.bitbltbuf.dpsm, m_transfer.bitbltbuf.dbp,
                                       std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u),
                                       x0, x0 + static_cast<int>(rrw) - 1, y0, y0 + static_cast<int>(rrh) - 1))
        return false;
    return !pages.Intersects(par->fbPages) && !pages.Intersects(par->zPages) && !pages.Intersects(par->texPages);
}

GSCpuBackend::~GSCpuBackend()
{
    StopWorker();
    m_par.reset();
}

const uint32_t *GSCpuBackend::GetPalette(const GSDrawState &state)
{
    const auto &tex = state.context.tex0;
    const bool fourBit = isFourBitIndexedPsm(tex.psm);
    const uint64_t key = static_cast<uint64_t>(tex.cpsm) |
                         (static_cast<uint64_t>(tex.csa) << 8) |
                         (static_cast<uint64_t>(fourBit ? 1u : 2u) << 16) |
                         (static_cast<uint64_t>(state.texa.ta0) << 24) |
                         (static_cast<uint64_t>(state.texa.aem ? 1u : 0u) << 32) |
                         (static_cast<uint64_t>(state.texa.ta1) << 40);
    if (key != m_paletteKey || m_paletteVersion != m_clutVersion)
    {
        const uint32_t count = fourBit ? 16u : 256u;
        for (uint32_t i = 0; i < count; ++i)
            m_palette[i] = LookupCLUT(state, static_cast<uint8_t>(i), tex.cpsm, tex.csm, tex.csa, tex.psm);
        if (fourBit)
            for (uint32_t i = 16; i < 256u; ++i)
                m_palette[i] = m_palette[i & 0xFu];
        m_paletteKey = key;
        m_paletteVersion = m_clutVersion;
    }
    return m_palette.data();
}

bool GSCpuBackend::DrawTriangleFast(const GSPrimitiveBatch &batch)
{
    return GSFastRaster::Draw<false>(*this, batch);
}

bool GSCpuBackend::DrawSpriteFast(const GSPrimitiveBatch &batch)
{
    return GSFastRaster::Draw<true>(*this, batch);
}
