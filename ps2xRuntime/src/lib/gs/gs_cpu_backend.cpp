#include <string>
#include <set>
#include <cstdlib>
#include <ostream>
#include <mutex>
#include <map>
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_capture.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_psmct16.h"
#include "runtime/gs/ps2_gs_psmct32.h"
#include "runtime/gs/ps2_gs_psmt4.h"
#include "runtime/gs/ps2_gs_psmt8.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "ps2_log.h"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace GSInternal;

namespace
{
    u16 Rgba8888ToRgba5551(u32 c)
    {
        uint32_t r = ((c >> 0) & 0xFF) >> 3;
        uint32_t g = ((c >> 8) & 0xFF) >> 3;
        uint32_t b = ((c >> 16) & 0xFF) >> 3;
        uint32_t a = ((c >> 24) & 0xFF) >> 7;

        return (r | (g << 5) | (b << 10) | (a << 15));
    }

    u32 Rgba5551ToRgba8888(u16 c)
    {
        u32 r = ((c >> 0) & 0x1F) << 3;
        u32 g = ((c >> 5) & 0x1F) << 3;
        u32 b = ((c >> 10) & 0x1F) << 3;
        u32 a = ((c >> 15) & 0x01) << 7;

        return (r | (g << 8) | (b << 16) | (a << 24));
    }

    u32 pack32(u8 r, u8 g, u8 b, u8 a)
    {
        return static_cast<u32>(r) | (g << 8) | (b << 16) | (a << 24);
    }

    uint32_t applyTexa(const GSTexaReg &texa, uint8_t psm, uint32_t texel)
    {
        if (psm == GS_PSM_CT32)
            return texel;

        const uint8_t r = static_cast<uint8_t>(texel & 0xFFu);
        const uint8_t g = static_cast<uint8_t>((texel >> 8) & 0xFFu);
        const uint8_t b = static_cast<uint8_t>((texel >> 16) & 0xFFu);
        const bool rgbZero = r == 0u && g == 0u && b == 0u;
        uint8_t a = static_cast<uint8_t>((texel >> 24) & 0xFFu);

        switch (psm)
        {
        case GS_PSM_CT24:
            a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            if ((a & 0x80u) != 0u)
                a = texa.ta1;
            else
                a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        default:
            break;
        }

        return (texel & 0x00FFFFFFu) | (static_cast<uint32_t>(a) << 24);
    }

    uint32_t addrPSMCT16Family(uint32_t basePtr, uint32_t width, uint8_t psm, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
            return GSPSMCT16::addrPSMCT16(basePtr, width, x, y);
        case GS_PSM_CT16S:
            return GSPSMCT16::addrPSMCT16S(basePtr, width, x, y);
        case GS_PSM_Z16:
            return GSPSMCT16::addrPSMZ16(basePtr, width, x, y);
        case GS_PSM_Z16S:
            return GSPSMCT16::addrPSMZ16S(basePtr, width, x, y);
        default:
            return 0u;
        }
    }

    std::atomic<uint32_t> s_debugPrimitiveCount{0};
    std::atomic<uint32_t> s_debugPixelCount{0};
    std::atomic<uint32_t> s_debugContext1PrimitiveCount{0};
    std::atomic<uint32_t> s_debugFbp150PixelCount{0};

    int wrapTextureCoordinate(int coordinate,
                              int textureSize,
                              uint8_t mode,
                              uint16_t regionMin,
                              uint16_t regionMax)
    {
        switch (mode & 0x3u)
        {
        case 0: // REPEAT
            return static_cast<int>(static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(textureSize - 1));
        case 1: // CLAMP
            return clampInt(coordinate, 0, textureSize - 1);
        case 2: // REGION_CLAMP
            return std::min(std::max(coordinate, static_cast<int>(regionMin)), static_cast<int>(regionMax));
        case 3: // REGION_REPEAT
            return static_cast<int>((static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(regionMin)) | static_cast<uint32_t>(regionMax));
        default:
            return coordinate;
        }
    }

    bool passesAlphaTest(uint64_t testReg, uint8_t alpha)
    {
        if ((testReg & 0x1u) == 0u)
            return true;

        const uint8_t atst = static_cast<uint8_t>((testReg >> 1) & 0x7u);
        const uint8_t aref = static_cast<uint8_t>((testReg >> 4) & 0xFFu);

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
        case 7:
            return alpha != aref;
        default:
            return true;
        }
    }

    struct PixelWriteMask
    {
        bool writeRgb = true;
        bool writeAlpha = true;
        bool writeDepth = true;

        bool writesFramebuffer() const
        {
            return writeRgb || writeAlpha;
        }

        bool writesAnything() const
        {
            return writesFramebuffer() || writeDepth;
        }
    };

    PixelWriteMask classifyAlphaTest(uint64_t testReg, uint8_t alpha, uint8_t framePsm)
    {
        const bool pass = passesAlphaTest(testReg, alpha);
        if (pass)
            return {};

        // TEST.AFAIL controls what happens when the alpha comparison fails.
        switch (static_cast<uint8_t>((testReg >> 12) & 0x3u))
        {
        case 1: // FB_ONLY
            return {true, true, false};
        case 2: // ZB_ONLY
            return {false, false, true};
        case 3: // RGB_ONLY
            // RGB_ONLY is only distinct for RGBA32. The GS treats it as
            // FB_ONLY for RGB24 and RGBA16 framebuffers.
            if (framePsm == GS_PSM_CT32)
                return {true, false, false};
            return {true, true, false};
        case 0: // KEEP
        default:
            return {false, false, false};
        }
    }

    bool passesDestinationAlphaTest(uint64_t testReg, uint8_t framePsm, uint32_t rawFramebufferPixel)
    {
        const bool date = ((testReg >> 14) & 0x1u) != 0u;
        if (!date)
            return true;

        const bool datm = ((testReg >> 15) & 0x1u) != 0u;
        switch (framePsm)
        {
        case GS_PSM_CT32:
            return (((rawFramebufferPixel >> 31) & 0x1u) != 0u) == datm;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            return (((rawFramebufferPixel >> 15) & 0x1u) != 0u) == datm;
        case GS_PSM_CT24:
            // RGB24 has no destination alpha, so DATE always passes.
            return true;
        default:
            return true;
        }
    }

    struct TextureCombineResult
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    TextureCombineResult combineTexture(const GSTex0Reg &tex,
                                        uint8_t vr,
                                        uint8_t vg,
                                        uint8_t vb,
                                        uint8_t va,
                                        uint8_t tr,
                                        uint8_t tg,
                                        uint8_t tb,
                                        uint8_t ta)
    {
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
        case 1: // DECAL
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
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
        default:
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        }

        return out;
    }

    uint32_t swizzleClutIndexCSM1(uint32_t index)
    {
        // CSM1 swaps address bits 3 and 4. Preserve the remaining bits:
        // 16-bit CLUTs expose a ninth address bit through CSA[4].
        return (index & ~0x18u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
    }

    bool isFourBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
    }

    bool isEightBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T8 || psm == GS_PSM_T8H;
    }

}

namespace
{
    static constexpr uint32_t kDefaultDisplayWidth = 640u;
    static constexpr uint32_t kDefaultDisplayHeight = 448u;
    static constexpr uint32_t kHostFrameWidth = 640u;
    static constexpr uint32_t kHostFrameHeight = 512u;

    uint16_t encodeFramePixelPSMCT16(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        return static_cast<uint16_t>(((r >> 3) & 0x1Fu) |
                                     (((g >> 3) & 0x1Fu) << 5) |
                                     (((b >> 3) & 0x1Fu) << 10) |
                                     ((a & 0x80u) ? 0x8000u : 0u)); // RGBA5551 keeps alpha bit 7, like every other GS write
    }

    void decodeDisplaySize(uint64_t display64, uint32_t &outWidth, uint32_t &outHeight)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);

        outWidth = (dw + 1u) / (magh + 1u);
        outHeight = dh + 1u;
        if (outWidth < 64u || outHeight < 64u)
        {
            outWidth = kDefaultDisplayWidth;
            outHeight = kDefaultDisplayHeight;
        }
        outWidth = std::min<uint32_t>(outWidth, kHostFrameWidth);
        outHeight = std::min<uint32_t>(outHeight, kHostFrameHeight);
    }

    GSFrameReg decodeDisplayFrame(uint64_t dispfb64)
    {
        GSFrameReg frame{};
        frame.fbp = static_cast<uint32_t>(dispfb64 & 0x1FFu);
        frame.fbw = static_cast<uint32_t>((dispfb64 >> 9) & 0x3Fu);
        frame.psm = static_cast<uint8_t>((dispfb64 >> 15) & 0x1Fu);
        return frame;
    }

    struct GSDisplayReadOrigin
    {
        uint32_t x = 0u;
        uint32_t y = 0u;
    };

    GSDisplayReadOrigin decodeDisplayReadOrigin(uint64_t dispfb64)
    {
        return {
            static_cast<uint32_t>((dispfb64 >> 32) & 0x7FFu),
            static_cast<uint32_t>((dispfb64 >> 43) & 0x7FFu)};
    }

    bool hasDisplaySetup(uint64_t display64, const GSFrameReg &frame)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);
        return frame.fbw != 0u || dw != 0u || dh != 0u || magh != 0u;
    }

    struct GSPmodeState
    {
        bool enableCrt1 = false;
        bool enableCrt2 = false;
        bool mmod = false;
        bool amod = false;
        bool slbg = false;
        uint8_t alp = 0u;
    };

    GSPmodeState decodePmode(uint64_t pmode64)
    {
        return {
            (pmode64 & 0x1ull) != 0ull,
            (pmode64 & 0x2ull) != 0ull,
            ((pmode64 >> 5) & 0x1ull) != 0ull,
            ((pmode64 >> 6) & 0x1ull) != 0ull,
            ((pmode64 >> 7) & 0x1ull) != 0ull,
            static_cast<uint8_t>((pmode64 >> 8) & 0xFFu)};
    }

    struct GSSmode2State
    {
        bool interlaced = false;
        bool frameMode = true;
    };

    GSSmode2State decodeSMode2(uint64_t smode2)
    {
        return {(smode2 & 0x1ull) != 0ull, ((smode2 >> 1) & 0x1ull) != 0ull};
    }

    void applyFieldPresentation(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height, bool oddField)
    {
        if (pixels.empty() || width == 0u || height < 2u)
            return;
        const std::vector<uint8_t> source = pixels;
        for (uint32_t y = 0; y < height; ++y)
        {
            uint32_t sourceY = ((y >> 1u) << 1u) + (oddField ? 1u : 0u);
            if (sourceY >= height)
                sourceY = height - 1u;
            std::memcpy(pixels.data() + y * kHostFrameWidth * 4u,
                        source.data() + sourceY * kHostFrameWidth * 4u,
                        width * 4u);
        }
    }

    void normalizePresentationAlpha(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        for (uint32_t y = 0; y < height; ++y)
        {
            uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
                row[x * 4u + 3u] = 255u;
        }
    }

    uint8_t blendPresentationChannel(uint8_t src, uint8_t dst, uint32_t factor)
    {
        const int delta = static_cast<int>(src) - static_cast<int>(dst);
        return GSInternal::clampU8(static_cast<int>(dst) + ((delta * static_cast<int>(factor)) / 255));
    }

    uint32_t countNonBlackPixels(const std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        uint32_t count = 0u;
        for (uint32_t y = 0; y < height; ++y)
        {
            const uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
            {
                if (row[x * 4u] != 0u || row[x * 4u + 1u] != 0u || row[x * 4u + 2u] != 0u)
                    ++count;
            }
        }
        return count;
    }
}

GSCpuBackend::GSCpuBackend()
{
    using namespace GSMem;
    static std::once_flag lookupTablesOnce;
    std::call_once(lookupTablesOnce, []()
                   { InitLookupTables(); });
    for (size_t i = 0; i < kPsmHandlerCount; ++i)
    {
        switch (i)
        {
        case GS_PSM_CT32:
            m_readVramFuncs[i] = ReadCT32;
            m_writeVramFuncs[i] = WriteCT32;
            break;
        case GS_PSM_CT24:
            m_readVramFuncs[i] = ReadCT24;
            m_writeVramFuncs[i] = WriteCT24;
            break;
        case GS_PSM_CT16:
            m_readVramFuncs[i] = ReadCT16;
            m_writeVramFuncs[i] = WriteCT16;
            break;
        case GS_PSM_CT16S:
            m_readVramFuncs[i] = ReadCT16S;
            m_writeVramFuncs[i] = WriteCT16S;
            break;
        case GS_PSM_T8:
            m_readVramFuncs[i] = ReadP8;
            m_writeVramFuncs[i] = WriteP8;
            break;
        case GS_PSM_T8H:
            m_readVramFuncs[i] = ReadP8H;
            m_writeVramFuncs[i] = WriteP8H;
            break;
        case GS_PSM_T4:
            m_readVramFuncs[i] = ReadP4;
            m_writeVramFuncs[i] = WriteP4;
            break;
        case GS_PSM_T4HH:
            m_readVramFuncs[i] = ReadP4HH;
            m_writeVramFuncs[i] = WriteP4HH;
            break;
        case GS_PSM_T4HL:
            m_readVramFuncs[i] = ReadP4HL;
            m_writeVramFuncs[i] = WriteP4HL;
            break;
        case GS_PSM_Z32:
            m_readVramFuncs[i] = ReadZ32;
            m_writeVramFuncs[i] = WriteZ32;
            break;
        case GS_PSM_Z24:
            m_readVramFuncs[i] = ReadZ24;
            m_writeVramFuncs[i] = WriteZ24;
            break;
        case GS_PSM_Z16:
            m_readVramFuncs[i] = ReadZ16;
            m_writeVramFuncs[i] = WriteZ16;
            break;
        case GS_PSM_Z16S:
            m_readVramFuncs[i] = ReadZ16S;
            m_writeVramFuncs[i] = WriteZ16S;
            break;
        default:
            m_readVramFuncs[i] = ReadNull;
            m_writeVramFuncs[i] = WriteNull;
            break;
        }
    }
    ResetUnlocked();
}

void GSCpuBackend::ExecReset()
{
    FlushParallel();
    if (CaptureTick())
        GSCapture::Writer::Instance().Record(GSCapture::kRecReset, nullptr, 0);
    ResetUnlocked();
}

void GSCpuBackend::ResetUnlocked()
{
    m_clut.fill(0u);
    ++m_clutVersion;
    m_clutCbp.fill(0u);
    m_texturePageCache.Invalidate();
    m_transfer = {};
    m_transfer.direction = 3u;
    m_transferState = {};
    m_transferState.direction = 3u;
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
}

bool GSCpuBackend::CaptureTick()
{
    GSCapture::Writer &w = GSCapture::Writer::Instance();
    if (!w.Enabled())
        return false;
    bool needState = false;
    if (!w.Tick(needState))
        return false;
    if (needState)
    {
        FlushParallel();
        const std::vector<uint8_t> state = SerializeStateUnlocked();
        w.Record(GSCapture::kRecState, state.data(), state.size());
    }
    return true;
}

namespace
{
    template <typename T>
    void appendPod(std::vector<uint8_t> &out, const T &v)
    {
        const size_t at = out.size();
        out.resize(at + sizeof(T));
        std::memcpy(out.data() + at, &v, sizeof(T));
    }

    template <typename T>
    bool readPod(const uint8_t *&p, const uint8_t *end, T &v)
    {
        if (static_cast<size_t>(end - p) < sizeof(T))
            return false;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return true;
    }
}

std::vector<uint8_t> GSCpuBackend::SerializeState() const
{
    WaitIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    const_cast<GSCpuBackend *>(this)->FlushParallel();
    return SerializeStateUnlocked();
}

std::vector<uint8_t> GSCpuBackend::SerializeStateUnlocked() const
{
    std::vector<uint8_t> out;
    out.reserve(m_vramSize + 32768u + m_localToHostBuffer.size());
    appendPod(out, m_vramSize);
    out.insert(out.end(), m_vram, m_vram + m_vramSize);
    appendPod(out, m_clut);
    appendPod(out, m_clutCbp);
    appendPod(out, m_texturePageCache.PageBase());
    const uint8_t *cacheBytes = m_texturePageCache.Bytes(m_vram);
    out.insert(out.end(), cacheBytes, cacheBytes + GSMem::TexturePageCache::kPageSize);
    appendPod(out, m_transfer);
    appendPod(out, m_transferState);
    appendPod(out, static_cast<uint64_t>(m_localToHostBuffer.size()));
    out.insert(out.end(), m_localToHostBuffer.begin(), m_localToHostBuffer.end());
    appendPod(out, static_cast<uint64_t>(m_localToHostReadPos));
    return out;
}

bool GSCpuBackend::RestoreState(const uint8_t *data, size_t size)
{
    WaitIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    const uint8_t *p = data;
    const uint8_t *end = data + size;
    uint32_t vramSize = 0;
    if (!readPod(p, end, vramSize) || vramSize != m_vramSize || static_cast<size_t>(end - p) < vramSize)
        return false;
    std::memcpy(m_vram, p, vramSize);
    p += vramSize;
    uint32_t pageBase = 0;
    if (!readPod(p, end, m_clut) || !readPod(p, end, m_clutCbp) || !readPod(p, end, pageBase) ||
        static_cast<size_t>(end - p) < GSMem::TexturePageCache::kPageSize)
        return false;
    m_texturePageCache.Restore(pageBase, p);
    ++m_clutVersion;
    p += GSMem::TexturePageCache::kPageSize;
    uint64_t l2hSize = 0, l2hPos = 0;
    if (!readPod(p, end, m_transfer) || !readPod(p, end, m_transferState) || !readPod(p, end, l2hSize) ||
        static_cast<uint64_t>(end - p) < l2hSize)
        return false;
    m_localToHostBuffer.assign(p, p + l2hSize);
    p += l2hSize;
    if (!readPod(p, end, l2hPos))
        return false;
    m_localToHostReadPos = static_cast<size_t>(l2hPos);
    return true;
}

void GSCpuBackend::ExecSubmit(const GSPrimitiveBatch &batch)
{
    if (CaptureTick())
        GSCapture::Writer::Instance().Record(GSCapture::kRecSubmit, &batch, sizeof(batch));
    if (!m_vram || batch.vertexCount == 0u)
        return;
    DrawPrimitive(batch);
}

void GSCpuBackend::ExecLoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    if (CaptureTick())
        GSCapture::Writer::Instance().Record(GSCapture::kRecLoadClut, &tex0, sizeof(tex0), &texclut, sizeof(texclut));
    if (!m_vram || (!isFourBitIndexedPsm(tex0.psm) && !isEightBitIndexedPsm(tex0.psm)))
        return;

    switch (tex0.cld)
    {
    case 0u:
    case 6u:
    case 7u:
        return;
    case 1u:
        break;
    case 2u:
        m_clutCbp[0] = tex0.cbp;
        break;
    case 3u:
        m_clutCbp[1] = tex0.cbp;
        break;
    case 4u:
        if (m_clutCbp[0] == tex0.cbp)
            return;
        m_clutCbp[0] = tex0.cbp;
        break;
    case 5u:
        if (m_clutCbp[1] == tex0.cbp)
            return;
        m_clutCbp[1] = tex0.cbp;
        break;
    default:
        return;
    }

    // The CLUT is read through the texture page cache: run it after the pending parallel
    // raster batch unless it provably observes the same VRAM bytes and cache state now.
    if (!DeferClutLoad(tex0, texclut))
        FlushParallel();
    LoadClutUnlocked(tex0, texclut);
}

void GSCpuBackend::LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    const bool fourBit = isFourBitIndexedPsm(tex0.psm);
    const bool sixteenBit = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
    const bool thirtyTwoBit = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
    if (!sixteenBit && !thirtyTwoBit)
        return;

    const uint32_t entryCount = fourBit ? 16u : 256u;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t destinationBase = (static_cast<uint32_t>(tex0.csa) & csaMask) << 4u;

    const bool loadCsm1Suffix = tex0.csm == 0u && thirtyTwoBit && !fourBit;
    const uint32_t firstEntry = loadCsm1Suffix ? destinationBase : 0u;

    for (uint32_t entry = firstEntry; entry < entryCount; ++entry)
    {
        uint32_t sourceX = 0u;
        uint32_t sourceY = 0u;
        uint32_t sourceWidth = 1u;

        if (tex0.csm == 0u)
        {
            const uint32_t sourceIndex = swizzleClutIndexCSM1(entry);
            sourceX = sourceIndex & 0x0Fu;
            sourceY = sourceIndex >> 4u;
        }
        else
        {
            sourceWidth = texclut.cbw != 0u ? static_cast<uint32_t>(texclut.cbw) : 1u;
            sourceX = (static_cast<uint32_t>(texclut.cou) << 4u) + entry;
            sourceY = static_cast<uint32_t>(texclut.cov);
        }

        const uint32_t raw = ReadTextureVramUnlocked(tex0.cpsm, tex0.cbp, sourceWidth, sourceX, sourceY);
        ++m_clutVersion;
        const uint32_t destination = (loadCsm1Suffix ? entry : destinationBase + entry) & (sixteenBit ? 0x1FFu : 0x0FFu);
        if (sixteenBit)
        {
            m_clut[destination] = static_cast<uint16_t>(raw);
        }
        else
        {
            m_clut[destination] = static_cast<uint16_t>(raw & 0xFFFFu);
            m_clut[destination + 256u] = static_cast<uint16_t>(raw >> 16u);
        }
    }
}

void GSCpuBackend::ExecTextureFlush()
{
    if (CaptureTick())
        GSCapture::Writer::Instance().Record(GSCapture::kRecTextureFlush, nullptr, 0);
    // Only affects the texture page cache: no need to finish the pending raster batch.
    DeferTextureFlush();
    m_texturePageCache.Invalidate();
}

uint32_t GSCpuBackend::ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    if (!m_vram)
        return 0u;
    return m_readVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y);
}

uint32_t GSCpuBackend::ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
{
    if (!m_vram)
        return 0u;

    return GSMem::ReadTexture(m_texturePageCache, m_vram, psm, base, bw, x, y);
}

void GSCpuBackend::ExecWriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    FlushParallel();
    if (CaptureTick())
    {
        const uint32_t args[6] = {psm, base, bw, x, y, value};
        GSCapture::Writer::Instance().Record(GSCapture::kRecWriteVram, args, sizeof(args));
    }
    WriteVramUnlocked(psm, base, bw, x, y, value);
}

void GSCpuBackend::WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    if (!m_vram)
        return;
    m_texturePageCache.PrepareUnknownWrite(m_vram);
    m_writeVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y, value);
}

void GSCpuBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    WaitIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    const_cast<GSCpuBackend *>(this)->FlushParallel();
    if (!m_vram || m_vramSize == 0u)
    {
        out.clear();
        return;
    }
    out.resize(m_vramSize);
    std::memcpy(out.data(), m_vram, m_vramSize);
}

GSTransferSnapshot GSCpuBackend::GetTransferSnapshot() const
{
    WaitIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    GSTransferSnapshot result = m_transferState;
    result.localToHostPendingBytes = m_localToHostReadPos < m_localToHostBuffer.size()
                                         ? m_localToHostBuffer.size() - m_localToHostReadPos
                                         : 0u;
    return result;
}

// DIAG: primitives drawn per (FBP, PSM, prim type) since last dump; read by the scheduler thread dump.
std::mutex g_gsDrawStatsMutex;
std::map<uint64_t, uint64_t> g_gsDrawStats;
void gsDumpDrawStats(std::ostream &out)
{
    std::lock_guard<std::mutex> lk(g_gsDrawStatsMutex);
    out << "[ee-gsdraw]";
    for (const auto &kv : g_gsDrawStats)
        out << " fbp=0x" << std::hex << (kv.first >> 16) << "/psm=0x" << ((kv.first >> 8) & 0xff) << "/prim=" << std::dec << (kv.first & 0xff) << ":" << kv.second;
    out << "\n";
    g_gsDrawStats.clear();
}

static char g_lastXfer2a00[96] = "none";

void GSCpuBackend::DrawPrimitive(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const auto &ctx = state.context;
    static const bool drawStats = std::getenv("PS2X_GS_DRAW_STATS") != nullptr;
    if (drawStats)
    {
        std::lock_guard<std::mutex> lk(g_gsDrawStatsMutex);
        ++g_gsDrawStats[(static_cast<uint64_t>(ctx.frame.fbp) << 16) | (static_cast<uint64_t>(ctx.frame.psm) << 8) | static_cast<uint8_t>(state.prim.type)];
    }
    static const bool primLog = std::getenv("PS2X_GS_PRIM_LOG") != nullptr;
    if (primLog && state.prim.tme)
    {
        FlushParallel();
        // One line per distinct (tex psm, clut psm, TEST, ALPHA, prim type, tfx, tcc, fbp) combination.
        static std::mutex m; static std::set<std::string> seen;
        char key[256];
        std::snprintf(key, sizeof(key), "prim=%u fbp=0x%x tex(psm=0x%x cpsm=0x%x csm=%u tcc=%u tfx=%u tbw=%u tw=%u th=%u) test=0x%llx alpha=0x%llx abe=%u fst=%u",
                      (unsigned)state.prim.type, (unsigned)ctx.frame.fbp, (unsigned)ctx.tex0.psm, (unsigned)ctx.tex0.cpsm, (unsigned)ctx.tex0.csm,
                      (unsigned)ctx.tex0.tcc, (unsigned)ctx.tex0.tfx, (unsigned)ctx.tex0.tbw, (unsigned)ctx.tex0.tw, (unsigned)ctx.tex0.th,
                      (unsigned long long)ctx.test, (unsigned long long)ctx.alpha, (unsigned)state.prim.abe, (unsigned)state.prim.fst);
        std::lock_guard<std::mutex> lk(m);
        // PS2X_GS_DUMP_TEX: write each distinct indexed texture (tbp0) decoded through the CLUT as a PPM.
        static const char *dumpDir = std::getenv("PS2X_GS_DUMP_TEX");
        static std::set<uint32_t> dumped;
        if (dumpDir && dumped.size() < 24 && (ctx.tex0.psm == GS_PSM_T4 || ctx.tex0.psm == GS_PSM_T8) &&
            dumped.insert(ctx.tex0.tbp0).second)
        {
            const int w = 1 << ctx.tex0.tw, h = 1 << ctx.tex0.th;
            char path[512];
            std::snprintf(path, sizeof(path), "%s/tex_%05x_psm%x_%dx%d.ppm", dumpDir, (unsigned)ctx.tex0.tbp0, (unsigned)ctx.tex0.psm, w, h);
            if (FILE *f = std::fopen(path, "wb"))
            {
                std::fprintf(f, "P6\n%d %d\n255\n", w * 2, h);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w * 2; ++x)
                    {
                        const int tx = x % w;
                        const uint32_t idx = ReadTextureVramUnlocked(ctx.tex0.psm, ctx.tex0.tbp0, ctx.tex0.tbw, tx, y);
                        const uint32_t c = LookupCLUT(state, static_cast<uint8_t>(idx), ctx.tex0.cpsm, ctx.tex0.csm, ctx.tex0.csa, ctx.tex0.psm);
                        // left half: RGB, right half: alpha as grey
                        const uint8_t a = static_cast<uint8_t>(std::min<uint32_t>(255u, (c >> 24) * 2u));
                        const uint8_t px[3] = {x < w ? (uint8_t)(c & 0xff) : a, x < w ? (uint8_t)((c >> 8) & 0xff) : a, x < w ? (uint8_t)((c >> 16) & 0xff) : a};
                        std::fwrite(px, 1, 3, f);
                    }
                std::fclose(f);
                // Raw upload-order CT32 view of the first page (what a 64x32 CT32 BITBLT wrote).
                std::string raw = std::string(path) + ".ct32.bin";
                if (FILE *rf = std::fopen(raw.c_str(), "wb"))
                {
                    for (uint32_t yy = 0; yy < 32u; ++yy)
                        for (uint32_t xx = 0; xx < 64u; ++xx)
                        {
                            const uint32_t word = ReadTextureVramUnlocked(GS_PSM_CT32, ctx.tex0.tbp0, 1u, xx, yy);
                            std::fwrite(&word, 4, 1, rf);
                        }
                    std::fclose(rf);
                }
                {
                    std::string cl;
                    for (uint32_t i = 0; i < 16u; ++i)
                    {
                        char e[16];
                        std::snprintf(e, sizeof(e), " %08x", LookupCLUT(state, static_cast<uint8_t>(i), ctx.tex0.cpsm, ctx.tex0.csm, ctx.tex0.csa, ctx.tex0.psm));
                        cl += e;
                    }
                    std::fprintf(stderr, "[gs-tex] clut cbp=0x%x cld=%u csa=%u csm=%u:%s\n", (unsigned)ctx.tex0.cbp, (unsigned)ctx.tex0.cld, (unsigned)ctx.tex0.csa, (unsigned)ctx.tex0.csm, cl.c_str());
                }
                std::fprintf(stderr, "[gs-tex] dumped %s (last upload to 0x2a00: %s)\n", path, g_lastXfer2a00);
            }
        }
        if (seen.size() < 200 && seen.insert(key).second)
            std::fprintf(stderr, "[gs-prim] %s texa=(ta0=%u aem=%u ta1=%u) dthe=%llu pabe=%u linear=%u\n", key, (unsigned)state.texa.ta0, (unsigned)state.texa.aem, (unsigned)state.texa.ta1, (unsigned long long)state.dthe, (unsigned)state.pabe, (unsigned)state.linearFilter);
    }
    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t primitiveIndex = s_debugPrimitiveCount.fetch_add(1u, std::memory_order_relaxed);
        if (primitiveIndex < 64u)
        {
            std::cout << "[gs:prim] idx=" << primitiveIndex
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tw=" << static_cast<uint32_t>(ctx.tex0.tw)
                      << " th=" << static_cast<uint32_t>(ctx.tex0.th)
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec
                      << " v0=(" << batch.vertices[0].x << "," << batch.vertices[0].y << ")"
                      << " uv0=(" << (batch.vertices[0].u >> 4) << "," << (batch.vertices[0].v >> 4) << ")"
                      << " stq0=(" << batch.vertices[0].s << "," << batch.vertices[0].t << "," << batch.vertices[0].q << ")"
                      << " v1=(" << batch.vertices[1].x << "," << batch.vertices[1].y << ")"
                      << " uv1=(" << (batch.vertices[1].u >> 4) << "," << (batch.vertices[1].v >> 4) << ")"
                      << " stq1=(" << batch.vertices[1].s << "," << batch.vertices[1].t << "," << batch.vertices[1].q << ")"
                      << " v2=(" << batch.vertices[2].x << "," << batch.vertices[2].y << ")"
                      << " uv2=(" << (batch.vertices[2].u >> 4) << "," << (batch.vertices[2].v >> 4) << ")"
                      << " stq2=(" << batch.vertices[2].s << "," << batch.vertices[2].t << "," << batch.vertices[2].q << ")"
                      << " rgba0=(" << static_cast<uint32_t>(batch.vertices[0].r) << ","
                      << static_cast<uint32_t>(batch.vertices[0].g) << ","
                      << static_cast<uint32_t>(batch.vertices[0].b) << ","
                      << static_cast<uint32_t>(batch.vertices[0].a) << ")"
                      << " rgba1=(" << static_cast<uint32_t>(batch.vertices[1].r) << ","
                      << static_cast<uint32_t>(batch.vertices[1].g) << ","
                      << static_cast<uint32_t>(batch.vertices[1].b) << ","
                      << static_cast<uint32_t>(batch.vertices[1].a) << ")"
                      << " rgba2=(" << static_cast<uint32_t>(batch.vertices[2].r) << ","
                      << static_cast<uint32_t>(batch.vertices[2].g) << ","
                      << static_cast<uint32_t>(batch.vertices[2].b) << ","
                      << static_cast<uint32_t>(batch.vertices[2].a) << ")"
                      << std::endl;
        }
    });

    PS2_IF_AGRESSIVE_LOGS({
        if ((state.prim.ctxt != 0u || ctx.frame.fbp == 150u) &&
            s_debugContext1PrimitiveCount.fetch_add(1u, std::memory_order_relaxed) < 32u)
        {
            std::cout << "[gs:copy-prim]"
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec << std::endl;
        }
    });

    static const bool slowPath = std::getenv("PS2X_GS_SLOW") != nullptr;
    switch (state.prim.type)
    {
    case GS_PRIM_SPRITE:
        if (slowPath || !DrawSpriteFast(batch))
        {
            FlushParallel();
            DrawSprite(batch);
        }
        break;
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
        if (slowPath || !DrawTriangleFast(batch))
        {
            FlushParallel();
            DrawTriangle(batch);
        }
        break;
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        FlushParallel();
        DrawLine(batch);
        break;
    case GS_PRIM_POINT:
        FlushParallel();
        DrawPoint(batch);
        break;
    default:
        break;
    }
}

void GSCpuBackend::WritePixel(const GSDrawState &state, int x, int y, uint32_t z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
{
    const auto &ctx = state.context;
    if (x < ctx.scissor.x0 || x > ctx.scissor.x1 || y < ctx.scissor.y0 || y > ctx.scissor.y1)
        return;

    // SCANMSK: 2 = drawing of even lines prohibited, 3 = odd lines prohibited.
    const uint32_t scanmsk = static_cast<uint32_t>(state.scanmsk & 3u);
    if (scanmsk >= 2u && (static_cast<uint32_t>(y) & 1u) == (scanmsk & 1u))
        return;

    if (state.prim.fge)
    {
        // GS manual: C = (F * C + (0xFF - F) * FOGCOL) >> 8.
        const uint32_t inverseFog = 255u - fog;
        auto applyFog = [&](uint8_t input, uint8_t fogColor) -> uint8_t
        {
            return static_cast<uint8_t>(((static_cast<uint32_t>(fog) * input) >> 8) + ((inverseFog * fogColor) >> 8));
        };

        r = applyFog(r, state.fogR);
        g = applyFog(g, state.fogG);
        b = applyFog(b, state.fogB);
    }

    const u32 fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    const u32 fbw = std::max<u32>(ctx.frame.fbw, 1u);
    const u32 fpsm = ctx.frame.psm;
    const u32 zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
    const u32 zpsm = ctx.zbuf.psm;

    // DATE on a 24-bit frame: hardware draws nothing at all (GSdx GSRendererSW::GetScanlineGlobalData).
    const bool date = ((ctx.test >> 14) & 0x1u) != 0u;
    if (date && (fpsm & 0xFu) == GS_PSM_CT24)
        return;

    PixelWriteMask writeMask = classifyAlphaTest(ctx.test, a, static_cast<uint8_t>(fpsm));

    // TEST.ZTE=0 disables both the depth test and depth writes (GSdx: zm = all ones, ztst = ALWAYS).
    const bool zte = ((ctx.test >> 16) & 0x1u) != 0u;
    if (!zte || ctx.zbuf.zmask)
        writeMask.writeDepth = false;
    if (!writeMask.writesAnything())
        return;

    const uint32_t ztestMethod = zte ? static_cast<uint32_t>((ctx.test >> 17) & 3u) : 1u;
    const bool alphaBlendEnabled = state.prim.abe;
    const bool preserveDestinationAlpha = writeMask.writeRgb && !writeMask.writeAlpha && fpsm == GS_PSM_CT32;
    const bool destinationAlphaTestNeedsRead = date && (fpsm == GS_PSM_CT32 || fpsm == GS_PSM_CT16 || fpsm == GS_PSM_CT16S);

    // small optimization, avoid reading the framebuffer for simple draws
    const bool frmw = destinationAlphaTestNeedsRead || (writeMask.writesFramebuffer() && ((ctx.frame.fbmsk != 0) || alphaBlendEnabled || preserveDestinationAlpha));

    u32 rawFramebufferPixel = 0;
    u32 fbrgba = 0;
    if (frmw)
    {
        rawFramebufferPixel = ReadVramUnlocked(fpsm, fbp, fbw, x, y);
        fbrgba = rawFramebufferPixel;

        if (bitsPerPixel(fpsm) == 16)
        {
            fbrgba = Rgba5551ToRgba8888(fbrgba);
        }
        else if (fpsm == GS_PSM_CT24)
        {
            // The GS supplies 0x80 as destination alpha for RGB24 blending.
            fbrgba |= 0x80000000u;
        }
    }

    if (!passesDestinationAlphaTest(ctx.test, static_cast<uint8_t>(fpsm), rawFramebufferPixel))
    {
        return;
    }

    // Vertex Z saturates to the depth format's range before both the test and the write.
    const uint32_t zFormat = zpsm & 0x3u;
    const uint32_t zMax = 0xFFFFFFFFu >> (zFormat * 8u);
    if (z > zMax)
        z = zMax;

    bool zpass = false;
    switch (ztestMethod)
    {
    case 0:
        zpass = false;
        break;
    case 1:
        zpass = true;
        break;
    case 2:
        zpass = z >= ReadVramUnlocked(zpsm, zbp, fbw, x, y);
        break;
    case 3:
        zpass = z > ReadVramUnlocked(zpsm, zbp, fbw, x, y);
        break;
    }

    static const bool zAlways = std::getenv("PS2X_GS_ZALWAYS") != nullptr; // debug: disable depth rejection
    if (!zpass && !zAlways)
    {
        return;
    }

    if (writeMask.writesFramebuffer())
    {
        int ir = r;
        int ig = g;
        int ib = b;

        // PABE disables alpha blending when the source alpha MSB is clear.
        if (alphaBlendEnabled && !(state.pabe && (a & 0x80u) == 0u))
        {
            const int dr = static_cast<int>(fbrgba & 0xFFu);
            const int dg = static_cast<int>((fbrgba >> 8) & 0xFFu);
            const int db = static_cast<int>((fbrgba >> 16) & 0xFFu);
            const int da = static_cast<int>((fbrgba >> 24) & 0xFFu);

            const uint64_t alphaReg = ctx.alpha;
            const uint8_t asel = alphaReg & 3;
            const uint8_t bsel = (alphaReg >> 2) & 3;
            const uint8_t csel = (alphaReg >> 4) & 3;
            const uint8_t dsel = (alphaReg >> 6) & 3;
            const int fix = static_cast<int>((alphaReg >> 32) & 0xFF);

            auto pick = [](uint8_t sel, int cs, int cd) -> int
            {
                return sel == 0 ? cs : (sel == 1 ? cd : 0);
            };
            const int cAlpha = (csel == 0) ? a : (csel == 1) ? da : fix;

            // ((A - B) * C >> 7) + D, with an arithmetic shift; the result is NOT clamped here:
            // COLCLAMP decides between saturation and 8-bit wrap below.
            ir = (((pick(asel, ir, dr) - pick(bsel, ir, dr)) * cAlpha) >> 7) + pick(dsel, ir, dr);
            ig = (((pick(asel, ig, dg) - pick(bsel, ig, dg)) * cAlpha) >> 7) + pick(dsel, ig, dg);
            ib = (((pick(asel, ib, db) - pick(bsel, ib, db)) * cAlpha) >> 7) + pick(dsel, ib, db);
        }

        // Dithering only applies to 16-bit frame buffers; DIMX holds signed 3-bit offsets for RGB.
        if ((state.dthe & 1u) != 0u && bitsPerPixel(fpsm) == 16)
        {
            const uint32_t shift = ((static_cast<uint32_t>(y) & 3u) * 16u) + ((static_cast<uint32_t>(x) & 3u) * 4u);
            int dm = static_cast<int>((state.dimx >> shift) & 0x7u);
            if (dm & 4)
                dm -= 8;
            ir += dm;
            ig += dm;
            ib += dm;
        }

        if ((state.colclamp & 1u) != 0u)
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

        if (writeMask.writeAlpha && (ctx.fba & 0x1ull) != 0ull && ctx.frame.psm != GS_PSM_CT24)
        {
            a = static_cast<uint8_t>(a | 0x80u);
        }

        u32 pixel = pack32(static_cast<u8>(ir), static_cast<u8>(ig), static_cast<u8>(ib), a);

        if (ctx.frame.fbmsk != 0)
        {
            pixel = (pixel & ~ctx.frame.fbmsk) | (fbrgba & ctx.frame.fbmsk);
        }

        if (preserveDestinationAlpha)
        {
            pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
        }

        // format conversion
        if (bitsPerPixel(fpsm) == 16)
        {
            pixel = Rgba8888ToRgba5551(pixel);
        }

        WriteVramUnlocked(fpsm, fbp, fbw, x, y, pixel);
    }

    if (writeMask.writeDepth)
    {
        WriteVramUnlocked(zpsm, zbp, fbw, x, y, z);
    }
}

uint32_t GSCpuBackend::LookupCLUT(const GSDrawState &state,
                                  uint8_t index,
                                  uint8_t cpsm,
                                  uint8_t csm,
                                  uint8_t csa,
                                  uint8_t sourcePsm)
{
    const bool sixteenBit = cpsm == GS_PSM_CT16 || cpsm == GS_PSM_CT16S;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t clutBase = (static_cast<uint32_t>(csa) & csaMask) << 4u;
    const uint32_t sourceIndex = isFourBitIndexedPsm(sourcePsm)
                                     ? (static_cast<uint32_t>(index) & 0x0Fu)
                                     : static_cast<uint32_t>(index);

    uint32_t clutIndex = (clutBase + sourceIndex) & (sixteenBit ? 0x1FFu : 0x0FFu);
    (void)csm;
    if (!sixteenBit && isEightBitIndexedPsm(sourcePsm))
    {
        // T8 with a 32-bit CLUT: the CSA-offset block index saturates at the last block (240)
        // instead of wrapping, independent of CSM (GSdx GSClut::ReadCLUT_T32_I8).
        const uint32_t block = std::min((sourceIndex & 0xF0u) + clutBase, 240u);
        clutIndex = block + (sourceIndex & 0x0Fu);
    }

    switch (cpsm)
    {
    case GS_PSM_CT32:
    {
        const uint32_t raw = static_cast<uint32_t>(m_clut[clutIndex]) | (static_cast<uint32_t>(m_clut[clutIndex + 256u]) << 16u);
        return applyTexa(state.texa, cpsm, raw);
    }
    case GS_PSM_CT24:
    {
        const uint32_t raw = static_cast<uint32_t>(m_clut[clutIndex]) | (static_cast<uint32_t>(m_clut[clutIndex + 256u]) << 16u);
        return applyTexa(state.texa, cpsm, raw & 0x00FFFFFFu);
    }
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
        return applyTexa(state.texa, cpsm, Rgba5551ToRgba8888(m_clut[clutIndex]));
    default:
        break;
    }

    return 0xFFFF00FFu;
}

uint32_t GSCpuBackend::FetchTexel(const GSDrawState &state, uint32_t tbp, uint32_t tbw, int u, int v)
{
    const auto &tex = state.context.tex0;
    const u32 out = ReadTextureVramUnlocked(tex.psm, tbp, tbw, static_cast<uint32_t>(u), static_cast<uint32_t>(v));

    switch (tex.psm)
    {
    case GS_PSM_CT32:
    case GS_PSM_Z32:
    case GS_PSM_CT24:
    case GS_PSM_Z24:
        return applyTexa(state.texa, static_cast<uint8_t>(tex.psm == GS_PSM_Z24 ? GS_PSM_CT24 : (tex.psm == GS_PSM_Z32 ? GS_PSM_CT32 : tex.psm)), out);
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
    case GS_PSM_Z16:
    case GS_PSM_Z16S:
        return applyTexa(state.texa, GS_PSM_CT16, Rgba5551ToRgba8888(static_cast<u16>(out)));
    case GS_PSM_T8:
    case GS_PSM_T8H:
    case GS_PSM_T4:
    case GS_PSM_T4HL:
    case GS_PSM_T4HH:
        return LookupCLUT(state, static_cast<u8>(out), tex.cpsm, tex.csm, tex.csa, tex.psm);
    default:
        break;
    }

    return 0xFFFF00FFu;
}

namespace
{
    inline int32_t textureCoordToFixed(double texel)
    {
        // Texel coordinate -> 16.16 fixed point, truncating like GSdx's float->int conversion.
        // Saturate to +/-(2^15 - 2) texels so the integer math below can never overflow.
        constexpr double kLimit = 32766.0;
        if (!(texel == texel))
            texel = 0.0;
        texel = std::min(std::max(texel, -kLimit), kLimit);
        return static_cast<int32_t>(texel * 65536.0);
    }

    inline int lerpTexel4(int a, int b, int f)
    {
        // GSdx lerp16_4: a + (((b - a) * f) >> 4) with a 4-bit weight.
        return a + (((b - a) * f) >> 4);
    }
}

uint32_t GSCpuBackend::SampleTexture(const GSDrawState &state, int32_t u, int32_t v, float q)
{
    const auto &ctx = state.context;
    const auto &tex = ctx.tex0;
    const uint64_t tex1 = ctx.tex1;

    // ---- LOD / filter selection (TEX1) ----
    const uint32_t mxl = static_cast<uint32_t>((tex1 >> 2) & 0x7u);
    const bool mmag = ((tex1 >> 5) & 0x1u) != 0u;
    const uint32_t mmin = static_cast<uint32_t>((tex1 >> 6) & 0x7u);
    bool linear = state.linearFilter;
    int lodi = 0;
    int32_t lodf = 0; // 16-bit fraction for tri-linear blending
    bool trilinear = false;
    if (mxl != 0u)
    {
        // MXL == 0 ignores MMIN entirely (state.linearFilter carries MMAG). Otherwise the
        // filter depends on the LOD: LOD <= 0 magnifies (MMAG), LOD > 0 minifies (MMIN).
        const bool lcm = (tex1 & 1u) != 0u;
        const uint32_t l = static_cast<uint32_t>((tex1 >> 19) & 0x3u);
        int32_t k = static_cast<int32_t>((tex1 >> 32) & 0xFFFu);
        if (k & 0x800)
            k -= 0x1000;
        double lod = static_cast<double>(k) / 16.0;
        if (!lcm && !state.prim.fst)
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
                lodFixed += 0x8000; // *_MIPMAP_NEAREST rounds to the nearest level
                lodi = std::min(lodFixed >> 16, maxLevel);
            }
        }
    }

    const uint64_t clamp = ctx.clamp;
    const uint8_t wrapU = static_cast<uint8_t>(clamp & 0x3u);
    const uint8_t wrapV = static_cast<uint8_t>((clamp >> 2) & 0x3u);

    auto sampleLevel = [&](int level, int32_t uu, int32_t vv) -> uint32_t
    {
        uint32_t tbp = tex.tbp0;
        uint32_t tbw = tex.tbw;
        if (level > 0)
        {
            // MIPTBP1 holds levels 1-3, MIPTBP2 levels 4-6 (MTBA fills MIPTBP1 on TEX0 writes).
            const uint64_t m = level <= 3 ? ctx.miptbp1 : ctx.miptbp2;
            const uint32_t shift = static_cast<uint32_t>(((level - 1) % 3) * 20);
            tbp = static_cast<uint32_t>((m >> shift) & 0x3FFFu);
            tbw = static_cast<uint32_t>((m >> (shift + 14u)) & 0x3Fu);
        }
        const int texW = std::max(1, static_cast<int>(state.textureWidth) >> level);
        const int texH = std::max(1, static_cast<int>(state.textureHeight) >> level);
        const uint16_t minU = static_cast<uint16_t>(((clamp >> 4) & 0x3FFu) >> level);
        const uint16_t maxU = static_cast<uint16_t>(((clamp >> 14) & 0x3FFu) >> level);
        const uint16_t minV = static_cast<uint16_t>(((clamp >> 24) & 0x3FFu) >> level);
        const uint16_t maxV = static_cast<uint16_t>(((clamp >> 34) & 0x3FFu) >> level);

        uu >>= level;
        vv >>= level;
        if (!linear)
        {
            const int su = wrapTextureCoordinate(uu >> 16, texW, wrapU, minU, maxU);
            const int sv = wrapTextureCoordinate(vv >> 16, texH, wrapV, minV, maxV);
            return FetchTexel(state, tbp, tbw, su, sv);
        }

        uu -= 0x8000;
        vv -= 0x8000;
        const int fu = (uu >> 12) & 0xF;
        const int fv = (vv >> 12) & 0xF;
        const int u0 = wrapTextureCoordinate(uu >> 16, texW, wrapU, minU, maxU);
        const int u1 = wrapTextureCoordinate((uu >> 16) + 1, texW, wrapU, minU, maxU);
        const int v0 = wrapTextureCoordinate(vv >> 16, texH, wrapV, minV, maxV);
        const int v1 = wrapTextureCoordinate((vv >> 16) + 1, texH, wrapV, minV, maxV);
        const uint32_t c00 = FetchTexel(state, tbp, tbw, u0, v0);
        const uint32_t c01 = FetchTexel(state, tbp, tbw, u1, v0);
        const uint32_t c10 = FetchTexel(state, tbp, tbw, u0, v1);
        const uint32_t c11 = FetchTexel(state, tbp, tbw, u1, v1);
        uint32_t result = 0u;
        for (uint32_t sh = 0; sh < 32u; sh += 8u)
        {
            const int top = lerpTexel4(static_cast<int>((c00 >> sh) & 0xFFu), static_cast<int>((c01 >> sh) & 0xFFu), fu);
            const int bottom = lerpTexel4(static_cast<int>((c10 >> sh) & 0xFFu), static_cast<int>((c11 >> sh) & 0xFFu), fu);
            result |= static_cast<uint32_t>(lerpTexel4(top, bottom, fv) & 0xFF) << sh;
        }
        return result;
    };

    const uint32_t c = sampleLevel(lodi, u, v);
    if (!trilinear)
        return c;

    // *_MIPMAP_LINEAR: blend with the next level by the LOD fraction (GSdx: lerp16<0> with a 15-bit weight).
    const uint32_t c2 = sampleLevel(lodi + 1, u, v);
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

namespace
{
    // Vertex position in 1/16 pixel units relative to the context's XYOFFSET (both 12.4 fixed point).
    inline int32_t vertexFixedX(const GSVertex &v, const GSContext &ctx)
    {
        return static_cast<int32_t>(std::lround(static_cast<double>(v.x) * 16.0)) - static_cast<int32_t>(ctx.xyoffset.ofx);
    }

    inline int32_t vertexFixedY(const GSVertex &v, const GSContext &ctx)
    {
        return static_cast<int32_t>(std::lround(static_cast<double>(v.y) * 16.0)) - static_cast<int32_t>(ctx.xyoffset.ofy);
    }

    inline int32_t ceilFixed4(int32_t value)
    {
        // ceil(value / 16) for 12.4 fixed point, correct for negative values (arithmetic shift).
        return (value + 15) >> 4;
    }

    inline uint32_t vertexZ(double z)
    {
        if (!(z > 0.0))
            return 0u;
        if (z >= 4294967295.0)
            return 0xFFFFFFFFu;
        return static_cast<uint32_t>(z);
    }

    // Interpolated attribute that stays exact when all three vertices agree.
    inline double interpolate3(double a0, double a1, double a2, double b1, double b2)
    {
        return a0 + (a1 - a0) * b1 + (a2 - a0) * b2;
    }

    inline int truncateAttribute(double value)
    {
        // GSdx converts interpolated attributes with truncation; the epsilon absorbs FP error
        // on values that are mathematically integral (e.g. vertex positions).
        return static_cast<int>(std::floor(value + 1.0e-6));
    }
}

void GSCpuBackend::DrawSprite(const GSPrimitiveBatch &batch)
{
    // GSdx GSRasterizer::DrawSprite: the covered pixels are [ceil(x0), ceil(x1)) x [ceil(y0), ceil(y1))
    // sampled at integer pixel positions; texture coordinates interpolate linearly between the two
    // vertices; colour, Z and fog come from the second vertex, and STQ uses the second vertex's Q.
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
    // Texture coordinates of each corner in level-0 texels (STQ divided by the second vertex's Q).
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
    int right = ceilFixed4(px1);   // exclusive
    int top = ceilFixed4(py0);
    int bottom = ceilFixed4(py1);  // exclusive
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
            for (int x = left; x < right; ++x)
                WritePixel(state, x, y, z, r, g, b, a, second.fog);
        return;
    }

    const double dudx = (px1 != px0) ? (tu1 - tu0) * 16.0 / static_cast<double>(px1 - px0) : 0.0;
    const double dvdy = (py1 != py0) ? (tv1 - tv0) * 16.0 / static_cast<double>(py1 - py0) : 0.0;
    const double ox = static_cast<double>(px0) / 16.0;
    const double oy = static_cast<double>(py0) / 16.0;
    const float q = static_cast<float>(q1);

    for (int y = top; y < bottom; ++y)
    {
        const int32_t vf = textureCoordToFixed(tv0 + dvdy * (static_cast<double>(y) - oy));
        for (int x = left; x < right; ++x)
        {
            const int32_t uf = textureCoordToFixed(tu0 + dudx * (static_cast<double>(x) - ox));
            const uint32_t texel = SampleTexture(state, uf, vf, q);
            const TextureCombineResult color = combineTexture(ctx.tex0, r, g, b, a,
                                                              static_cast<uint8_t>(texel), static_cast<uint8_t>(texel >> 8),
                                                              static_cast<uint8_t>(texel >> 16), static_cast<uint8_t>(texel >> 24));
            WritePixel(state, x, y, z, color.r, color.g, color.b, color.a, second.fog);
        }
    }
}

void GSCpuBackend::DrawTriangle(const GSPrimitiveBatch &batch)
{
    // Top-left fill convention with integer sample points, matching GSdx GSRasterizer::DrawTriangle
    // (rows [ceil(ytop), ceil(ybottom)), spans [ceil(xleft), ceil(xright))). Shared edges of strips
    // and fans are therefore drawn exactly once, which matters for alpha-blended geometry.
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

    // Edge i is opposite vertex i: E_i(P) = (B - A) x (P - A) with A = v[i+1], B = v[i+2].
    int64_t ex[3], ey[3], ec[3];
    bool topLeft[3];
    for (int i = 0; i < 3; ++i)
    {
        const int a = (i + 1) % 3;
        const int b = (i + 2) % 3;
        // E(P) = (Xb - Xa) * (Py - Ya) - (Yb - Ya) * (Px - Xa) = ex * Px + ey * Py + ec
        ex[i] = -(Y[b] - Y[a]) * sign;
        ey[i] = (X[b] - X[a]) * sign;
        ec[i] = (-(X[b] - X[a]) * Y[a] + (Y[b] - Y[a]) * X[a]) * sign;
        // Interior is where E > 0. A pixel exactly on the edge belongs to it when the interior lies
        // to its right (left edge) or, for horizontal edges, below it (top edge).
        topLeft[i] = ex[i] > 0 || (ex[i] == 0 && ey[i] > 0);
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

    for (int y = minY; y <= maxY; ++y)
    {
        const int64_t py = static_cast<int64_t>(y) * 16;
        for (int x = minX; x <= maxX; ++x)
        {
            const int64_t px = static_cast<int64_t>(x) * 16;
            int64_t w[3];
            bool inside = true;
            for (int i = 0; i < 3 && inside; ++i)
            {
                w[i] = ex[i] * px + ey[i] * py + ec[i];
                inside = w[i] > 0 || (w[i] == 0 && topLeft[i]);
            }
            if (!inside)
                continue;

            const double b1 = static_cast<double>(w[1]) * invArea;
            const double b2 = static_cast<double>(w[2]) * invArea;

            const uint32_t z = vertexZ(std::floor(interpolate3(v0.z, v1.z, v2.z, b1, b2) + 1.0e-6));

            uint8_t r, g, b, a;
            if (state.prim.iip)
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

            if (state.prim.tme)
            {
                int32_t uf, vf;
                float q = 1.0f;
                if (state.prim.fst)
                {
                    uf = textureCoordToFixed(interpolate3(v0.u, v1.u, v2.u, b1, b2) / 16.0);
                    vf = textureCoordToFixed(interpolate3(v0.v, v1.v, v2.v, b1, b2) / 16.0);
                }
                else
                {
                    // The GS DDA interpolates the homogeneous S, T and Q values; texel
                    // coordinates are S/Q and T/Q after interpolation.
                    const double is = interpolate3(v0.s, v1.s, v2.s, b1, b2);
                    const double it = interpolate3(v0.t, v1.t, v2.t, b1, b2);
                    double iq = interpolate3(v0.q, v1.q, v2.q, b1, b2);
                    if (iq == 0.0)
                        iq = 1.0e-30;
                    uf = textureCoordToFixed(is / iq * texW);
                    vf = textureCoordToFixed(it / iq * texH);
                    q = static_cast<float>(iq);
                }

                const uint32_t texel = SampleTexture(state, uf, vf, q);
                const TextureCombineResult color = combineTexture(ctx.tex0, r, g, b, a,
                                                                  static_cast<uint8_t>(texel), static_cast<uint8_t>(texel >> 8),
                                                                  static_cast<uint8_t>(texel >> 16), static_cast<uint8_t>(texel >> 24));
                r = color.r;
                g = color.g;
                b = color.b;
                a = color.a;
            }

            const uint8_t fog = clampU8(truncateAttribute(interpolate3(v0.fog, v1.fog, v2.fog, b1, b2)));
            WritePixel(state, x, y, z, r, g, b, a, fog);
        }
    }
}

#include "gs_cpu_fast.inl"
#include "gs_cpu_async.inl"

void GSCpuBackend::DrawPoint(const GSPrimitiveBatch &batch)
{
    // GSdx GSRasterizer::DrawPoint: the pixel containing (x + 0.5, y + 0.5).
    const GSDrawState &state = batch.state;
    const auto &ctx = state.context;
    const GSVertex &v = batch.vertices[0];
    const int x = (vertexFixedX(v, ctx) + 8) >> 4;
    const int y = (vertexFixedY(v, ctx) + 8) >> 4;
    uint8_t r = v.r, g = v.g, b = v.b, a = v.a;
    if (state.prim.tme)
    {
        int32_t uf, vf;
        float q = 1.0f;
        if (state.prim.fst)
        {
            uf = textureCoordToFixed(v.u / 16.0);
            vf = textureCoordToFixed(v.v / 16.0);
        }
        else
        {
            const double iq = v.q != 0.0f ? v.q : 1.0e-30;
            uf = textureCoordToFixed(v.s / iq * state.textureWidth);
            vf = textureCoordToFixed(v.t / iq * state.textureHeight);
            q = v.q;
        }
        const uint32_t texel = SampleTexture(state, uf, vf, q);
        const TextureCombineResult color = combineTexture(ctx.tex0, r, g, b, a,
                                                          static_cast<uint8_t>(texel), static_cast<uint8_t>(texel >> 8),
                                                          static_cast<uint8_t>(texel >> 16), static_cast<uint8_t>(texel >> 24));
        r = color.r;
        g = color.g;
        b = color.b;
        a = color.a;
    }
    WritePixel(state, x, y, vertexZ(v.z), r, g, b, a, v.fog);
}

void GSCpuBackend::DrawLine(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int x0 = (vertexFixedX(v0, ctx) + 8) >> 4;
    int y0 = (vertexFixedY(v0, ctx) + 8) >> 4;
    const int x1 = (vertexFixedX(v1, ctx) + 8) >> 4;
    const int y1 = (vertexFixedY(v1, ctx) + 8) >> 4;

    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    int totalSteps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    if (totalSteps == 0)
        totalSteps = 1;
    int step = 0;

    for (;;)
    {
        const double t = static_cast<double>(step) / static_cast<double>(totalSteps);
        auto lerp = [t](double a, double b) { return a + (b - a) * t; };
        uint8_t r, g, b, a;
        if (state.prim.iip)
        {
            r = clampU8(truncateAttribute(lerp(v0.r, v1.r)));
            g = clampU8(truncateAttribute(lerp(v0.g, v1.g)));
            b = clampU8(truncateAttribute(lerp(v0.b, v1.b)));
            a = clampU8(truncateAttribute(lerp(v0.a, v1.a)));
        }
        else
        {
            r = v1.r;
            g = v1.g;
            b = v1.b;
            a = v1.a;
        }

        if (state.prim.tme)
        {
            int32_t uf, vf;
            float q = 1.0f;
            if (state.prim.fst)
            {
                uf = textureCoordToFixed(lerp(v0.u, v1.u) / 16.0);
                vf = textureCoordToFixed(lerp(v0.v, v1.v) / 16.0);
            }
            else
            {
                double iq = lerp(v0.q, v1.q);
                if (iq == 0.0)
                    iq = 1.0e-30;
                uf = textureCoordToFixed(lerp(v0.s, v1.s) / iq * state.textureWidth);
                vf = textureCoordToFixed(lerp(v0.t, v1.t) / iq * state.textureHeight);
                q = static_cast<float>(iq);
            }
            const uint32_t texel = SampleTexture(state, uf, vf, q);
            const TextureCombineResult color = combineTexture(ctx.tex0, r, g, b, a,
                                                              static_cast<uint8_t>(texel), static_cast<uint8_t>(texel >> 8),
                                                              static_cast<uint8_t>(texel >> 16), static_cast<uint8_t>(texel >> 24));
            r = color.r;
            g = color.g;
            b = color.b;
            a = color.a;
        }

        const uint32_t z = vertexZ(std::floor(lerp(v0.z, v1.z) + 1.0e-6));
        const uint8_t fog = clampU8(truncateAttribute(lerp(v0.fog, v1.fog)));
        WritePixel(state, x0, y0, z, r, g, b, a, fog);

        if (x0 == x1 && y0 == y1)
            break;

        int e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
        ++step;
    }
}

void GSCpuBackend::ExecBeginTransfer(const GSTransferCommand &command)
{
    // Host->local only records the transfer here (no VRAM or texture cache access); the
    // uploads decide whether the pending raster batch must be drawn first (DeferUpload).
    if (command.direction != 0u)
        FlushParallel();
    if (CaptureTick())
        GSCapture::Writer::Instance().Record(GSCapture::kRecBeginTransfer, &command, sizeof(command));
    m_transfer = command;
    m_transferState.x = command.trxpos.dsax;
    m_transferState.y = command.trxpos.dsay;
    m_transferState.totalPixels = static_cast<uint32_t>(command.trxreg.rrw) * static_cast<uint32_t>(command.trxreg.rrh);
    m_transferState.copiedPixels = 0u;
    m_transferState.direction = command.direction;
    m_transferState.localToHostPendingBytes = 0u;
    static const bool xferLog = std::getenv("PS2X_GS_PRIM_LOG") != nullptr;
    static int xferCount = 0;
    if (command.bitbltbuf.dbp == 0x2a00u)
        std::snprintf(g_lastXfer2a00, sizeof(g_lastXfer2a00), "#%d dpsm=0x%x rr=%ux%u", xferCount, (unsigned)command.bitbltbuf.dpsm, (unsigned)command.trxreg.rrw, (unsigned)command.trxreg.rrh);
    if (xferLog && xferCount++ < 400)
        std::fprintf(stderr, "[gs-xfer] dir=%u dbp=0x%x dbw=%u dpsm=0x%x sbp=0x%x sbw=%u spsm=0x%x dsa=(%u,%u) ssa=(%u,%u) rr=%ux%u dir2=%u\n",
                     (unsigned)command.direction, (unsigned)command.bitbltbuf.dbp, (unsigned)command.bitbltbuf.dbw, (unsigned)command.bitbltbuf.dpsm,
                     (unsigned)command.bitbltbuf.sbp, (unsigned)command.bitbltbuf.sbw, (unsigned)command.bitbltbuf.spsm,
                     (unsigned)command.trxpos.dsax, (unsigned)command.trxpos.dsay, (unsigned)command.trxpos.ssax, (unsigned)command.trxpos.ssay,
                     (unsigned)command.trxreg.rrw, (unsigned)command.trxreg.rrh, (unsigned)command.trxpos.dir);

    if (command.direction == 2u)
        PerformLocalToLocalTransfer();
    else if (command.direction == 1u)
        PerformLocalToHostTransfer();
}

void GSCpuBackend::ExecUploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    if (!DeferUpload())
        FlushParallel();
    if (data && CaptureTick())
        GSCapture::Writer::Instance().Record(GSCapture::kRecUpload, data, sizeBytes);
    if (!data || sizeBytes == 0u || !m_vram || m_transferState.direction != 0u)
        return;
    if (m_transfer.trxreg.rrw == 0u || m_transfer.trxreg.rrh == 0u || m_transferState.totalPixels == 0u)
        return;

    const uint32_t dbp = m_transfer.bitbltbuf.dbp;
    const uint32_t dbw = std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u);
    const uint8_t dpsm = m_transfer.bitbltbuf.dpsm;
    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t dsax = m_transfer.trxpos.dsax;
    uint32_t offset = 0u;

    auto advancePixel = [&](uint32_t count)
    {
        const uint32_t totalPixels = m_transferState.totalPixels;
        m_transferState.copiedPixels =
            std::min<uint32_t>(totalPixels, m_transferState.copiedPixels + count);

        if (m_transferState.copiedPixels >= totalPixels)
        {
            m_transferState.direction = 3u;
            m_transferState.totalPixels = 0u;
            return;
        }

        m_transferState.x = dsax + (m_transferState.copiedPixels % rrw);
        m_transferState.y = m_transfer.trxpos.dsay + (m_transferState.copiedPixels / rrw);
    };

    while (offset < sizeBytes && m_transferState.direction == 0u)
    {
        switch (dpsm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        {
            if (sizeBytes - offset < 4u)
                return;
            uint32_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 4u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT24:
        case GS_PSM_Z24:
        {
            if (sizeBytes - offset < 3u)
                return;
            const uint32_t value = static_cast<uint32_t>(data[offset]) |
                                   (static_cast<uint32_t>(data[offset + 1u]) << 8u) |
                                   (static_cast<uint32_t>(data[offset + 2u]) << 16u);
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 3u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
        {
            if (sizeBytes - offset < 2u)
                return;
            uint16_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 2u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_T8:
        case GS_PSM_T8H:
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, data[offset++]);
            advancePixel(1u);
            break;
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
        {
            const uint8_t packed = data[offset++];
            const uint32_t firstPixel = m_transferState.copiedPixels;
            WriteVramUnlocked(dpsm, dbp, dbw,
                              dsax + (firstPixel % rrw),
                              m_transfer.trxpos.dsay + (firstPixel / rrw),
                              packed & 0x0Fu);
            if (firstPixel + 1u < m_transferState.totalPixels)
            {
                const uint32_t secondPixel = firstPixel + 1u;
                WriteVramUnlocked(dpsm, dbp, dbw,
                                  dsax + (secondPixel % rrw),
                                  m_transfer.trxpos.dsay + (secondPixel / rrw),
                                  (packed >> 4u) & 0x0Fu);
            }
            advancePixel(std::min<uint32_t>(2u, m_transferState.totalPixels - firstPixel));
            break;
        }
        default:
            return;
        }
    }
}

void GSCpuBackend::PerformLocalToLocalTransfer()
{
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t total = rrw * rrh;
    if (total == 0u)
    {
        m_transferState.direction = 3u;
        return;
    }

    for (uint32_t pixel = 0; pixel < total; ++pixel)
    {
        uint32_t x = pixel % rrw;
        uint32_t y = pixel / rrw;
        if ((m_transfer.trxpos.dir & 0x2u) != 0u)
            x = rrw - x - 1u;
        if ((m_transfer.trxpos.dir & 0x1u) != 0u)
            y = rrh - y - 1u;

        const uint32_t value = ReadVramUnlocked(m_transfer.bitbltbuf.spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u),
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        WriteVramUnlocked(m_transfer.bitbltbuf.dpsm,
                          m_transfer.bitbltbuf.dbp,
                          std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u),
                          x + m_transfer.trxpos.dsax,
                          y + m_transfer.trxpos.dsay,
                          value);
    }

    m_transferState.copiedPixels = total;
    m_transferState.direction = 3u;
}

void GSCpuBackend::PerformLocalToHostTransfer()
{
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t sbw = std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u);
    const uint8_t spsm = m_transfer.bitbltbuf.spsm;
    const uint32_t bpp = static_cast<uint32_t>(GSMem::BitsPerPixel(static_cast<GSMem::PixelStorageMode>(spsm)));
    const uint32_t total = rrw * rrh;
    m_localToHostBuffer.reserve((static_cast<size_t>(total) * bpp + 7u) / 8u);

    for (uint32_t pixel = 0u; pixel < total; ++pixel)
    {
        const uint32_t x = pixel % rrw;
        const uint32_t y = pixel / rrw;
        const uint32_t value = ReadVramUnlocked(spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                sbw,
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        switch (bpp)
        {
        case 32:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 24u));
            break;
        case 24:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            break;
        case 16:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            break;
        case 8:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            break;
        case 4:
        {
            if ((pixel & 1u) != 0u)
                break;
            uint32_t next = 0u;
            if (pixel + 1u < total)
            {
                const uint32_t nextPixel = pixel + 1u;
                const uint32_t nextX = nextPixel % rrw;
                const uint32_t nextY = nextPixel / rrw;
                next = ReadVramUnlocked(spsm, m_transfer.bitbltbuf.sbp, sbw,
                                        nextX + m_transfer.trxpos.ssax,
                                        nextY + m_transfer.trxpos.ssay);
            }
            m_localToHostBuffer.push_back(static_cast<uint8_t>((value & 0x0Fu) | ((next & 0x0Fu) << 4u)));
            break;
        }
        default:
            break;
        }
    }

    m_transferState.copiedPixels = total;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size();
}

uint32_t GSCpuBackend::ExecConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    FlushParallel();
    if (dst && CaptureTick())
        GSCapture::Writer::Instance().Record(GSCapture::kRecConsume, &maxBytes, sizeof(maxBytes));
    if (!dst || maxBytes == 0u || m_localToHostReadPos >= m_localToHostBuffer.size())
        return 0u;
    const size_t count = std::min<size_t>(maxBytes, m_localToHostBuffer.size() - m_localToHostReadPos);
    std::memcpy(dst, m_localToHostBuffer.data() + m_localToHostReadPos, count);
    m_localToHostReadPos += count;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size() - m_localToHostReadPos;
    return static_cast<uint32_t>(count);
}

bool GSCpuBackend::ExecClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    FlushParallel();
    if (CaptureTick())
        GSCapture::Writer::Instance().Record(GSCapture::kRecClear, &context, sizeof(context), &rgba, sizeof(rgba));
    if (!m_vram || context.frame.fbw == 0u)
        return false;

    const uint32_t x0 = context.scissor.x0;
    const uint32_t x1 = std::max<uint32_t>(x0, context.scissor.x1);
    const uint32_t y0 = context.scissor.y0;
    const uint32_t y1 = std::max<uint32_t>(y0, context.scissor.y1);
    uint8_t r = static_cast<uint8_t>(rgba);
    uint8_t g = static_cast<uint8_t>(rgba >> 8u);
    uint8_t b = static_cast<uint8_t>(rgba >> 16u);
    uint8_t a = static_cast<uint8_t>(rgba >> 24u);
    if ((context.fba & 1ull) != 0ull && context.frame.psm != GS_PSM_CT24)
        a |= 0x80u;

    const uint32_t fbp = GSInternal::framePageBaseToBlock(context.frame.fbp);
    const uint32_t fbw = std::max<uint32_t>(context.frame.fbw, 1u);
    if (context.frame.psm == GS_PSM_CT32 || context.frame.psm == GS_PSM_CT24)
    {
        const uint32_t source = static_cast<uint32_t>(r) |
                                (static_cast<uint32_t>(g) << 8u) |
                                (static_cast<uint32_t>(b) << 16u) |
                                (static_cast<uint32_t>(a) << 24u);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint32_t pixel = source;
                if (context.frame.fbmsk != 0u)
                {
                    const uint32_t old = ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y);
                    pixel = (pixel & ~context.frame.fbmsk) | (old & context.frame.fbmsk);
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return true;
    }

    if (context.frame.psm == GS_PSM_CT16 || context.frame.psm == GS_PSM_CT16S)
    {
        const uint16_t source = encodeFramePixelPSMCT16(r, g, b, a);
        const uint16_t mask = static_cast<uint16_t>(context.frame.fbmsk);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint16_t pixel = source;
                if (mask != 0u)
                {
                    const uint16_t old = static_cast<uint16_t>(ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y));
                    pixel = static_cast<uint16_t>((pixel & ~mask) | (old & mask));
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return true;
    }
    return false;
}

bool GSCpuBackend::CopyFrameToHostRgba(const GSFrameReg &frame,
                                       uint32_t width,
                                       uint32_t height,
                                       std::vector<uint8_t> &outPixels,
                                       bool preserveAlpha,
                                       bool useLocalMemoryLayout,
                                       bool frameBaseIsPages,
                                       uint32_t sourceOriginX,
                                       uint32_t sourceOriginY) const
{
    if (!m_vram || m_vramSize == 0u)
        return false;

    outPixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
    const uint32_t baseBytes = frameBaseIsPages ? frame.fbp * 8192u : frame.fbp * 256u;
    const uint32_t basePtr = frameBaseIsPages ? GSInternal::framePageBaseToBlock(frame.fbp) : frame.fbp;
    const uint32_t fbw = frame.fbw ? frame.fbw : kHostFrameWidth / 64u;
    const uint32_t bytesPerPixel = (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S) ? 2u : 4u;
    const uint32_t stride = fbw * 64u * bytesPerPixel;

    for (uint32_t y = 0; y < height; ++y)
    {
        uint8_t *dst = outPixels.data() + y * kHostFrameWidth * 4u;
        for (uint32_t x = 0; x < width; ++x)
        {
            const uint32_t sx = sourceOriginX + x;
            const uint32_t sy = sourceOriginY + y;
            if (frame.psm == GS_PSM_CT32 || frame.psm == GS_PSM_CT24)
            {
                uint32_t color = 0u;
                if (useLocalMemoryLayout)
                    color = ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy);
                else
                {
                    const uint32_t pixelBytes = frame.psm == GS_PSM_CT24 ? 3u : 4u;
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * pixelBytes;
                    if (offset + pixelBytes > m_vramSize)
                        return false;
                    color = m_vram[offset] | (static_cast<uint32_t>(m_vram[offset + 1u]) << 8u) |
                            (static_cast<uint32_t>(m_vram[offset + 2u]) << 16u);
                    if (pixelBytes == 4u)
                        color |= static_cast<uint32_t>(m_vram[offset + 3u]) << 24u;
                }
                dst[x * 4u] = static_cast<uint8_t>(color);
                dst[x * 4u + 1u] = static_cast<uint8_t>(color >> 8u);
                dst[x * 4u + 2u] = static_cast<uint8_t>(color >> 16u);
                dst[x * 4u + 3u] = preserveAlpha && frame.psm != GS_PSM_CT24 ? static_cast<uint8_t>(color >> 24u) : 255u;
            }
            else if (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S)
            {
                uint16_t color = 0u;
                if (useLocalMemoryLayout)
                    color = static_cast<uint16_t>(ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy));
                else
                {
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * 2u;
                    if (offset + 2u > m_vramSize)
                        return false;
                    std::memcpy(&color, m_vram + offset, sizeof(color));
                }
                const uint32_t r = color & 31u;
                const uint32_t g = (color >> 5u) & 31u;
                const uint32_t b = (color >> 10u) & 31u;
                dst[x * 4u] = static_cast<uint8_t>((r << 3u) | (r >> 2u));
                dst[x * 4u + 1u] = static_cast<uint8_t>((g << 3u) | (g >> 2u));
                dst[x * 4u + 2u] = static_cast<uint8_t>((b << 3u) | (b >> 2u));
                dst[x * 4u + 3u] = preserveAlpha ? ((color & 0x8000u) ? 0x80u : 0u) : 255u;
            }
            else
            {
                outPixels.clear();
                return false;
            }
        }
    }
    return true;
}

PresentationFrame GSCpuBackend::Present(const GSPresentationRequest &request)
{
    // Snapshot local memory under the backend lock, then perform the expensive
    // display conversion without holding the producer-side raster lock.
    thread_local std::vector<uint8_t> snapshot;
    WaitIdle();
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        FlushParallel();
        if (CaptureTick())
            GSCapture::Writer::Instance().Record(GSCapture::kRecPresent, &request, sizeof(request));
        if (!m_vram || m_vramSize == 0u)
            snapshot.clear();
        else
        {
            snapshot.resize(m_vramSize);
            std::memcpy(snapshot.data(), m_vram, m_vramSize);
        }
    }
    if (snapshot.empty())
        return {};

    thread_local GSCpuBackend snapshotBackend;
    snapshotBackend.Initialize(snapshot.data(), static_cast<uint32_t>(snapshot.size()));
    return snapshotBackend.PresentFromLocalMemory(request);
}

PresentationFrame GSCpuBackend::PresentFromLocalMemory(const GSPresentationRequest &request)
{
    PresentationFrame result{};
    const GSPmodeState pmode = decodePmode(request.pmode);
    const GSSmode2State smode2 = decodeSMode2(request.smode2);
    const bool fieldMode = smode2.interlaced && !smode2.frameMode;
    const bool oddField = (request.vsyncTick & 1ull) != 0ull;
    const GSFrameReg displayFrame1 = decodeDisplayFrame(request.dispfb1);
    const GSFrameReg displayFrame2 = decodeDisplayFrame(request.dispfb2);
    const GSDisplayReadOrigin origin1 = decodeDisplayReadOrigin(request.dispfb1);
    const GSDisplayReadOrigin origin2 = decodeDisplayReadOrigin(request.dispfb2);
    uint32_t width1 = 0u, height1 = 0u, width2 = 0u, height2 = 0u;
    decodeDisplaySize(request.display1, width1, height1);
    decodeDisplaySize(request.display2, width2, height2);
    const bool valid1 = pmode.enableCrt1 && hasDisplaySetup(request.display1, displayFrame1);
    const bool valid2 = pmode.enableCrt2 && hasDisplaySetup(request.display2, displayFrame2);
    if (!valid1 && !valid2)
        return result;

    auto copySource = [&](const GSFrameReg &displayFrame,
                          const GSDisplayReadOrigin &origin,
                          uint32_t width,
                          uint32_t height,
                          bool allowPreferred,
                          bool preserveAlpha,
                          GSFrameReg &selected,
                          std::vector<uint8_t> &pixels,
                          bool &usedPreferred) -> bool
    {
        selected = displayFrame;
        pixels.clear();
        usedPreferred = false;
        if (allowPreferred && request.hasPreferredSource && request.preferredDestFbp == displayFrame.fbp &&
            (request.preferredSource.fbw != 0u || request.preferredSource.fbp != displayFrame.fbp) &&
            CopyFrameToHostRgba(request.preferredSource, width, height, pixels, preserveAlpha, true, false, 0u, 0u))
        {
            selected = request.preferredSource;
            usedPreferred = true;
        }
        if (pixels.empty() && !CopyFrameToHostRgba(displayFrame, width, height, pixels, preserveAlpha, true, true, origin.x, origin.y))
            return false;

        if (!usedPreferred && displayFrame.fbp == 0u && countNonBlackPixels(pixels, width, height) == 0u)
        {
            for (const GSFrameReg &candidate : request.contextFrames)
            {
                if (candidate.fbp == selected.fbp && candidate.fbw == selected.fbw && candidate.psm == selected.psm)
                    continue;
                std::vector<uint8_t> candidatePixels;
                if (!CopyFrameToHostRgba(candidate, width, height, candidatePixels, preserveAlpha, true, true, 0u, 0u))
                    continue;
                if (countNonBlackPixels(candidatePixels, width, height) == 0u)
                    continue;
                selected = candidate;
                pixels.swap(candidatePixels);
                break;
            }
        }
        return true;
    };

    if (valid1 && valid2)
    {
        GSFrameReg selected1{}, selected2{};
        std::vector<uint8_t> crt1, crt2;
        bool preferred1 = false, preferred2 = false;
        if (copySource(displayFrame1, origin1, width1, height1, false, true, selected1, crt1, preferred1) &&
            copySource(displayFrame2, origin2, width2, height2, false, true, selected2, crt2, preferred2))
        {
            result.width = std::max(width1, width2);
            result.height = std::max(height1, height2);
            result.pixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
            const uint8_t bgR = static_cast<uint8_t>(request.bgcolor);
            const uint8_t bgG = static_cast<uint8_t>(request.bgcolor >> 8u);
            const uint8_t bgB = static_cast<uint8_t>(request.bgcolor >> 16u);
            for (uint32_t y = 0; y < result.height; ++y)
                for (uint32_t x = 0; x < result.width; ++x)
                {
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    dst[0] = bgR;
                    dst[1] = bgG;
                    dst[2] = bgB;
                    dst[3] = pmode.alp;
                }
            if (!pmode.slbg)
                for (uint32_t y = 0; y < height2; ++y)
                    std::memcpy(result.pixels.data() + y * kHostFrameWidth * 4u, crt2.data() + y * kHostFrameWidth * 4u, width2 * 4u);
            for (uint32_t y = 0; y < height1; ++y)
                for (uint32_t x = 0; x < width1; ++x)
                {
                    const uint8_t *src = crt1.data() + (y * kHostFrameWidth + x) * 4u;
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    const uint32_t factor = pmode.mmod ? pmode.alp : std::min<uint32_t>(255u, static_cast<uint32_t>(src[3]) * 2u);
                    dst[0] = blendPresentationChannel(src[0], dst[0], factor);
                    dst[1] = blendPresentationChannel(src[1], dst[1], factor);
                    dst[2] = blendPresentationChannel(src[2], dst[2], factor);
                    dst[3] = pmode.amod ? dst[3] : src[3];
                }
            normalizePresentationAlpha(result.pixels, result.width, result.height);
            if (fieldMode)
                applyFieldPresentation(result.pixels, result.width, result.height, oddField);
            result.displayFbp = displayFrame1.fbp;
            result.sourceFbp = selected1.fbp;
            return result;
        }
    }

    const GSFrameReg &displayFrame = valid1 ? displayFrame1 : displayFrame2;
    const GSDisplayReadOrigin &origin = valid1 ? origin1 : origin2;
    result.width = valid1 ? width1 : width2;
    result.height = valid1 ? height1 : height2;
    GSFrameReg selected = displayFrame;
    if (!copySource(displayFrame, origin, result.width, result.height, true, false, selected, result.pixels, result.usedPreferred))
        return {};
    if (fieldMode)
        applyFieldPresentation(result.pixels, result.width, result.height, oddField);
    normalizePresentationAlpha(result.pixels, result.width, result.height);
    result.displayFbp = displayFrame.fbp;
    result.sourceFbp = selected.fbp;
    return result;
}
