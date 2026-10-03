#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ostream>
#include <atomic>
// Based on Blackline Interactive implementation
#include "runtime/ps2_memory.h"
#include <cstring>

enum VIFCmd : uint8_t
{
    VIF_NOP = 0x00,
    VIF_STCYCL = 0x01,
    VIF_OFFSET = 0x02,
    VIF_BASE = 0x03,
    VIF_ITOP = 0x04,
    VIF_STMOD = 0x05,
    VIF_MSKPATH3 = 0x06,
    VIF_MARK = 0x07,
    VIF_FLUSHE = 0x10,
    VIF_FLUSH = 0x11,
    VIF_FLUSHA = 0x13,
    VIF_MSCAL = 0x14,
    VIF_MSCALF = 0x15,
    VIF_MSCNT = 0x17,
    VIF_STMASK = 0x20,
    VIF_STROW = 0x30,
    VIF_STCOL = 0x31,
    VIF_MPG = 0x4A,
    VIF_DIRECT = 0x50,
    VIF_DIRECTHL = 0x51,
};

namespace
{
    constexpr uint8_t kGifFmtImage = 2u;

    uint32_t pendingGifImageQwc(const uint8_t *data, uint32_t sizeBytes)
    {
        if (!data || sizeBytes < 16u)
            return 0u;

        uint32_t offset = 0u;
        while (offset + 16u <= sizeBytes)
        {
            uint64_t tagLo = 0u;
            std::memcpy(&tagLo, data + offset, sizeof(tagLo));
            offset += 16u;

            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t payloadBytes = 0u;
            if (flg == 0u) // PACKED
            {
                payloadBytes = static_cast<uint64_t>(nloop) * nreg * 16ull;
            }
            else if (flg == 1u) // REGLIST, padded to a quadword
            {
                payloadBytes = static_cast<uint64_t>(nloop) * nreg * 8ull;
                payloadBytes = (payloadBytes + 15ull) & ~15ull;
            }
            else if (flg == kGifFmtImage)
            {
                payloadBytes = static_cast<uint64_t>(nloop) * 16ull;
                const uint64_t availableBytes = sizeBytes - offset;
                if (payloadBytes > availableBytes)
                {
                    return static_cast<uint32_t>((payloadBytes - availableBytes) / 16ull);
                }
            }
            else
            {
                return 0u;
            }

            if (payloadBytes > static_cast<uint64_t>(sizeBytes - offset))
                return 0u;
            offset += static_cast<uint32_t>(payloadBytes);
        }

        return 0u;
    }
}

void PS2Memory::processVIF0Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    processVIF0Data(m_rdram + srcPhys, sizeBytes);
}

void PS2Memory::processVIF0Data(const uint8_t *data, uint32_t sizeBytes)
{
    if (sizeBytes == 0u)
        return;

    uint32_t pos = 0;
    while (pos + 4 <= sizeBytes)
    {
        uint32_t cmd = 0u;
        std::memcpy(&cmd, data + pos, sizeof(cmd));
        pos += 4u;

        const uint8_t opcode = static_cast<uint8_t>((cmd >> 24) & 0x7Fu);
        const uint16_t imm = static_cast<uint16_t>(cmd & 0xFFFFu);
        const uint8_t num = static_cast<uint8_t>((cmd >> 16) & 0xFFu);
        const bool irq = (cmd & 0x80000000u) != 0u;

        vif0_regs.code = cmd;
        vif0_regs.num = num;
        if (irq)
            vif0_regs.stat |= (1u << 11);

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif0_regs.cycle = imm;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            vif0_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif0_regs.mode = imm & 3u;
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif0_regs.mark = imm;
            vif0_regs.stat |= (1u << 6);
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4u > sizeBytes)
                break;
            std::memcpy(&vif0_regs.mask, data + pos, sizeof(vif0_regs.mask));
            pos += 4u;
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16u > sizeBytes)
                break;
            std::memcpy(vif0_regs.row, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16u > sizeBytes)
                break;
            std::memcpy(vif0_regs.col, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            const uint32_t destAddr = static_cast<uint32_t>(imm & 0x1FFu) * 8u;
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            uint32_t copyBytes = 0u;
            if (m_vu0Code && destAddr < PS2_VU0_CODE_SIZE && mpgBytes > 0u)
            {
                copyBytes = mpgBytes;
                if (destAddr + copyBytes > PS2_VU0_CODE_SIZE)
                    copyBytes = PS2_VU0_CODE_SIZE - destAddr;
                if (pos + copyBytes <= sizeBytes)
                {
                    std::memcpy(m_vu0Code + destAddr, data + pos, copyBytes);
                    markVU0CodeModified();
                }
            }

            pos += mpgBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else if ((opcode & 0x60u) == 0x60u)
        {
            const uint8_t vn = static_cast<uint8_t>((opcode >> 2) & 0x3u);
            const uint8_t vl = static_cast<uint8_t>(opcode & 0x3u);
            const int components = static_cast<int>(vn) + 1;
            int bitsPerComponent = 32;
            switch (vl)
            {
            case 0:
                bitsPerComponent = 32;
                break;
            case 1:
                bitsPerComponent = 16;
                break;
            case 2:
                bitsPerComponent = 8;
                break;
            case 3:
                bitsPerComponent = (vn == 3u) ? 4 : 16;
                break;
            default:
                break;
            }
            const int bitsPerVector = (vl == 3u && vn == 3u) ? 16 : (components * bitsPerComponent);
            uint32_t bytesPerVector = static_cast<uint32_t>((bitsPerVector + 7) / 8);
            const uint32_t writeVectorCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            uint32_t cl = vif0_regs.cycle & 0xFFu;
            uint32_t wl = (vif0_regs.cycle >> 8) & 0xFFu;
            if (cl == 0u)
                cl = 1u;
            if (wl == 0u)
                wl = 1u;
            uint32_t sourceVectorCount = writeVectorCount;
            if (cl < wl)
            {
                const uint32_t fullBlocks = writeVectorCount / wl;
                uint32_t remainder = writeVectorCount % wl;
                if (remainder > cl)
                    remainder = cl;
                sourceVectorCount = fullBlocks * cl + remainder;
            }
            uint32_t totalBytes = sourceVectorCount * bytesPerVector;
            totalBytes = (totalBytes + 3u) & ~3u;

            if (m_vu0Data && pos + totalBytes <= sizeBytes && vl == 0u)
            {
                uint32_t vuAddr = static_cast<uint32_t>(imm & 0x3FFu);
                if ((imm & 0x8000u) != 0u)
                    vuAddr = (vuAddr + (vif0_regs.tops & 0x3FFu)) & 0x3FFu;
                const uint8_t *srcBase = data + pos;
                uint32_t srcIndex = 0u;
                for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
                {
                    const uint32_t cyclePos = writeIndex % wl;
                    const bool sourceAvailable = (cl >= wl) || (cyclePos < cl);
                    uint32_t destVec = (cl >= wl) ? ((vuAddr + (writeIndex / wl) * cl + cyclePos) & 0x3FFu)
                                                  : ((vuAddr + writeIndex) & 0x3FFu);
                    const uint32_t destOff = destVec * 16u;
                    if (destOff + 16u > PS2_VU0_DATA_SIZE)
                    {
                        if (sourceAvailable && srcIndex < sourceVectorCount)
                            ++srcIndex;
                        continue;
                    }
                    if (!sourceAvailable || srcIndex >= sourceVectorCount)
                        continue;
                    const uint8_t *srcVec = srcBase + srcIndex * bytesPerVector;
                    ++srcIndex;
                    uint32_t lanes[4] = {0u, 0u, 0u, 0u};
                    std::memcpy(lanes, m_vu0Data + destOff, sizeof(lanes));
                    const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                    for (uint32_t c = 0; c < limit; ++c)
                    {
                        uint32_t scalar = 0u;
                        std::memcpy(&scalar, srcVec + c * 4u, sizeof(scalar));
                        lanes[c] = scalar;
                    }
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(m_vu0Data + destOff), _mm_loadu_si128(reinterpret_cast<const __m128i *>(lanes)));
                }
            }
            pos += totalBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else
        {
            break;
        }
    }
}

std::atomic<uint64_t> g_vif1OpcodeCounts[128];
void ps2xDumpVif1Stats(std::ostream &out)
{
    out << "[ee-vif1] opcodes:";
    for (int i = 0; i < 128; ++i) { uint64_t c = g_vif1OpcodeCounts[i].exchange(0); if (c) out << " 0x" << std::hex << i << std::dec << ":" << c; }
    out << "\n";
}

static constexpr uint32_t kVif1RawRing = 4u;
static std::vector<uint8_t> g_vif1RawRing[kVif1RawRing];
static uint32_t g_vif1RawHead = 0u;
struct Vif1TraceEntry { uint32_t cmd, tops, base, ofst, dbf, firstWord; };
static constexpr uint32_t kVif1TraceSize = 64u;
static Vif1TraceEntry g_vif1Trace[kVif1TraceSize];
static uint32_t g_vif1TraceHead = 0u;
void Ps2DumpVif1Trace(const char *why)
{
    if (const char *dir = std::getenv("PS2X_VIF1_FAILDUMP"); dir && dir[0])
    {
        for (uint32_t i = 0; i < kVif1RawRing; ++i)
        {
            const auto &buf = g_vif1RawRing[(g_vif1RawHead + i) % kVif1RawRing];
            char path[512];
            std::snprintf(path, sizeof(path), "%s/vif1_fail_%u.bin", dir, i);
            if (FILE *f = std::fopen(path, "wb")) { std::fwrite(buf.data(), 1, buf.size(), f); std::fclose(f); }
        }
        std::fprintf(stderr, "[vif1-trace] raw buffers written to %s (0 = oldest)\n", dir);
    }
    std::fprintf(stderr, "[vif1-trace] last %u VIF1 commands (%s), oldest first:\n", kVif1TraceSize, why);
    for (uint32_t i = 0; i < kVif1TraceSize; ++i)
    {
        const Vif1TraceEntry &e = g_vif1Trace[(g_vif1TraceHead + i) % kVif1TraceSize];
        std::fprintf(stderr, "  cmd=%08x op=%02x num=%3u imm=%04x | tops=%03x base=%03x ofst=%03x dbf=%u data0=%08x\n",
                     e.cmd, (e.cmd >> 24) & 0x7f, (e.cmd >> 16) & 0xff, e.cmd & 0xffff, e.tops, e.base, e.ofst, e.dbf, e.firstWord);
    }
}

// ---------------------------------------------------------------------------------------------
// VIF1 UNPACK fast path. Produces exactly the same VU1 memory and ROW register contents as the generic
// per-lane loop in processVIF1Data for the cases it accepts (no write mask, CL >= WL, decodable format):
//  * lanes beyond the format's component count keep their previous VU memory value, and in
//    offset/difference mode (MODE 1/2) ROW is still added to them (generic-loop behaviour);
//  * MODE 2 updates ROW after every vector; V4-5 never adds ROW.
// ---------------------------------------------------------------------------------------------
static int g_vu1WatchPrints = 0; // "[vu1-watch]" print budget (first 16 unpack writes to qw 0x3F4/0x3F5)

namespace
{
    template <uint32_t VN, uint32_t VL>
    inline void vif1DecodeVector(const uint8_t *s, bool zext, uint32_t v[4])
    {
        constexpr uint32_t comps = VN + 1u;
        if constexpr (VL == 0u)
        {
            if constexpr (VN == 0u)
            {
                uint32_t x;
                std::memcpy(&x, s, 4);
                v[0] = v[1] = v[2] = v[3] = x;
            }
            else
            {
                std::memcpy(v, s, comps * 4u);
            }
        }
        else if constexpr (VL == 1u)
        {
            auto ext = [zext](uint16_t r) -> uint32_t
            { return zext ? static_cast<uint32_t>(r) : static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(r))); };
            if constexpr (VN == 0u)
            {
                uint16_t r;
                std::memcpy(&r, s, 2);
                v[0] = v[1] = v[2] = v[3] = ext(r);
            }
            else
            {
                for (uint32_t c = 0; c < comps; ++c)
                {
                    uint16_t r;
                    std::memcpy(&r, s + c * 2u, 2);
                    v[c] = ext(r);
                }
            }
        }
        else if constexpr (VL == 2u)
        {
            auto ext = [zext](uint8_t r) -> uint32_t
            { return zext ? static_cast<uint32_t>(r) : static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(r))); };
            if constexpr (VN == 0u)
            {
                v[0] = v[1] = v[2] = v[3] = ext(s[0]);
            }
            else
            {
                for (uint32_t c = 0; c < comps; ++c)
                    v[c] = ext(s[c]);
            }
        }
        else // V4-5 (VN == 3, VL == 3)
        {
            uint16_t p;
            std::memcpy(&p, s, 2);
            v[0] = (p & 0x1Fu) << 3u;
            v[1] = ((p >> 5) & 0x1Fu) << 3u;
            v[2] = ((p >> 10) & 0x1Fu) << 3u;
            v[3] = ((p >> 15) & 0x01u) << 7u;
        }
    }

    template <uint32_t VN, uint32_t VL, uint32_t MODE>
    void vif1UnpackFastT(uint8_t *vu, uint32_t row[4], const uint8_t *src, uint32_t vuAddr, uint32_t count,
                         uint32_t cl, uint32_t wl, bool zext)
    {
        constexpr uint32_t bpv = (VL == 3u) ? 2u : ((VN + 1u) * (32u >> VL) / 8u);
        constexpr bool canAdd = (VL != 3u);
        uint32_t r0 = row[0], r1 = row[1], r2 = row[2], r3 = row[3];
        uint32_t block = 0u, cyc = 0u; // writeIndex / wl, writeIndex % wl
        for (uint32_t i = 0; i < count; ++i, src += bpv)
        {
            const uint32_t destVec = (vuAddr + block * cl + cyc) & 0x3FFu;
            if (++cyc == wl)
            {
                cyc = 0u;
                ++block;
            }
            uint8_t *d = vu + destVec * 16u;
            uint32_t v[4];
            if constexpr (VN != 3u && !(VN == 0u))
                std::memcpy(v, d, 16); // lanes past the component count keep their old value
            vif1DecodeVector<VN, VL>(src, zext, v);
            if constexpr (canAdd && MODE == 1u)
            {
                v[0] += r0; v[1] += r1; v[2] += r2; v[3] += r3;
            }
            else if constexpr (canAdd && MODE == 2u)
            {
                r0 = v[0] += r0; r1 = v[1] += r1; r2 = v[2] += r2; r3 = v[3] += r3;
            }
            std::memcpy(d, v, 16);
        }
        if constexpr (canAdd && MODE == 2u)
        {
            row[0] = r0; row[1] = r1; row[2] = r2; row[3] = r3;
        }
    }

    template <uint32_t VN, uint32_t VL>
    void vif1UnpackFastM(uint8_t *vu, uint32_t row[4], const uint8_t *src, uint32_t vuAddr, uint32_t count,
                         uint32_t cl, uint32_t wl, uint32_t mode, bool zext)
    {
        if (mode == 1u)
            vif1UnpackFastT<VN, VL, 1u>(vu, row, src, vuAddr, count, cl, wl, zext);
        else if (mode == 2u)
            vif1UnpackFastT<VN, VL, 2u>(vu, row, src, vuAddr, count, cl, wl, zext);
        else
            vif1UnpackFastT<VN, VL, 0u>(vu, row, src, vuAddr, count, cl, wl, zext);
    }

    void vif1UnpackFast(uint8_t *vu, uint32_t row[4], const uint8_t *src, uint32_t vuAddr, uint32_t count,
                        uint32_t cl, uint32_t wl, uint32_t vn, uint32_t vl, uint32_t mode, bool zext)
    {
        switch (vn * 4u + vl)
        {
        case 0: vif1UnpackFastM<0, 0>(vu, row, src, vuAddr, count, cl, wl, mode, zext); break;
        case 1: vif1UnpackFastM<0, 1>(vu, row, src, vuAddr, count, cl, wl, mode, zext); break;
        case 2: vif1UnpackFastM<0, 2>(vu, row, src, vuAddr, count, cl, wl, mode, zext); break;
        case 4: vif1UnpackFastM<1, 0>(vu, row, src, vuAddr, count, cl, wl, mode, zext); break;
        case 5: vif1UnpackFastM<1, 1>(vu, row, src, vuAddr, count, cl, wl, mode, zext); break;
        case 6: vif1UnpackFastM<1, 2>(vu, row, src, vuAddr, count, cl, wl, mode, zext); break;
        case 8: vif1UnpackFastM<2, 0>(vu, row, src, vuAddr, count, cl, wl, mode, zext); break;
        case 9: vif1UnpackFastM<2, 1>(vu, row, src, vuAddr, count, cl, wl, mode, zext); break;
        case 10: vif1UnpackFastM<2, 2>(vu, row, src, vuAddr, count, cl, wl, mode, zext); break;
        case 12: vif1UnpackFastM<3, 0>(vu, row, src, vuAddr, count, cl, wl, mode, zext); break;
        case 13: vif1UnpackFastM<3, 1>(vu, row, src, vuAddr, count, cl, wl, mode, zext); break;
        case 14: vif1UnpackFastM<3, 2>(vu, row, src, vuAddr, count, cl, wl, mode, zext); break;
        case 15: vif1UnpackFastM<3, 3>(vu, row, src, vuAddr, count, cl, wl, mode, zext); break;
        default: break; // vl == 3 with vn != 3 never reaches the fast path
        }
    }

    // True while the "[vu1-watch]" diagnostic could still print for this UNPACK (it then takes the
    // generic path so stderr output stays identical). Conservative bounding-range test.
    bool vif1UnpackTouchesWatch(uint32_t vuAddr, uint32_t count, uint32_t cl, uint32_t wl)
    {
        if (g_vu1WatchPrints >= 16 || count == 0u)
            return false;
        const uint32_t last = count - 1u;
        const uint32_t span = (last / wl) * cl + (last % wl);
        if (span >= 0x3FEu)
            return true;
        return ((0x3F4u - vuAddr) & 0x3FFu) <= span || ((0x3F5u - vuAddr) & 0x3FFu) <= span;
    }

    bool vif1VerifyEnabled()
    {
        static const bool on = []
        {
            const char *v = std::getenv("PS2X_VIF_VERIFY");
            return v && v[0] && v[0] != '0';
        }();
        return on;
    }

    struct Vif1VerifyStats
    {
        uint64_t checked = 0u, mismatched = 0u;
        ~Vif1VerifyStats()
        {
            if (checked)
                std::fprintf(stderr, "[vif-verify] final checked=%llu mismatched=%llu\n",
                             static_cast<unsigned long long>(checked), static_cast<unsigned long long>(mismatched));
        }
    };

    void vif1VerifyReport(uint8_t opcode, uint16_t imm, uint8_t num, uint32_t cycle, uint32_t mode, bool same)
    {
        static Vif1VerifyStats stats;
        uint64_t &checked = stats.checked, &mismatched = stats.mismatched;
        ++checked;
        if (!same)
        {
            ++mismatched;
            if (mismatched <= 32u)
                std::fprintf(stderr, "[vif-verify] MISMATCH op=0x%02x imm=0x%04x num=%u cycle=0x%04x mode=%u\n",
                             opcode, imm, num, cycle, mode);
        }
        if ((checked & ((1u << 18) - 1u)) == 0u)
            std::fprintf(stderr, "[vif-verify] checked=%llu mismatched=%llu\n",
                         static_cast<unsigned long long>(checked), static_cast<unsigned long long>(mismatched));
    }
}

void PS2Memory::processVIF1Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    processVIF1Data(m_rdram + srcPhys, sizeBytes);
}

// PATH2 data from DIRECT/DIRECTHL payloads. The GIF keeps IMAGE-mode state across packets: a game may
// send the IMAGE GIFtag in one DIRECT and its pixel data in following DIRECTs (SotC: DIRECT 1 [tag],
// NOP, DIRECT 0x800 [data]). Only DIRECT payloads feed the pending image, never raw VIF stream words.
void PS2Memory::feedVif1Path2(const uint8_t *payload, uint32_t qwCount, bool directHl)
{
    while (qwCount != 0u)
    {
        if (m_vif1PendingPath2ImageQwc != 0u)
        {
            const uint32_t chunkQw = std::min<uint32_t>(m_vif1PendingPath2ImageQwc, qwCount);
            std::vector<uint8_t> imagePacket(16u + static_cast<size_t>(chunkQw) * 16u, 0u);
            const uint64_t imageTag =
                static_cast<uint64_t>(chunkQw & 0x7FFFu) |
                ((m_vif1PendingPath2ImageQwc == chunkQw) ? (1ull << 15) : 0ull) |
                (static_cast<uint64_t>(kGifFmtImage) << 58);
            std::memcpy(imagePacket.data(), &imageTag, sizeof(imageTag));
            std::memcpy(imagePacket.data() + 16u, payload, static_cast<size_t>(chunkQw) * 16u);
            submitGifPacket(GifPathId::Path2, imagePacket.data(), static_cast<uint32_t>(imagePacket.size()), true, m_vif1PendingPath2DirectHl);
            payload += chunkQw * 16u;
            qwCount -= chunkQw;
            m_vif1PendingPath2ImageQwc -= chunkQw;
            if (m_vif1PendingPath2ImageQwc == 0u)
                m_vif1PendingPath2DirectHl = false;
            continue;
        }
        submitGifPacket(GifPathId::Path2, payload, qwCount * 16u, true, directHl);
        const uint32_t pendingImageQw = pendingGifImageQwc(payload, qwCount * 16u);
        if (pendingImageQw != 0u)
        {
            m_vif1PendingPath2ImageQwc = pendingImageQw;
            m_vif1PendingPath2DirectHl = directHl;
        }
        break;
    }
}

void PS2Memory::processVIF1Data(const uint8_t *data, uint32_t sizeBytes)
{
    if (sizeBytes == 0u)
        return;
    {   // DIAG: PS2X_DUMP_VIF1=file -> append [u32 size][bytes] records for offline decoding (first 400)
        static FILE *dump = [] { const char *p = std::getenv("PS2X_DUMP_VIF1"); return p ? std::fopen(p, "wb") : nullptr; }();
        static int records = 0, seen = 0;
        static const int skip = [] { const char *p = std::getenv("PS2X_DUMP_VIF1_SKIP"); return p ? std::atoi(p) : 0; }();
        if (dump && seen++ >= skip && records < 400) { std::fwrite(&sizeBytes, 4, 1, dump); std::fwrite(data, 1, sizeBytes, dump); std::fflush(dump); ++records; }
    }

    {   // keep the last few raw buffers so the first bad XGKICK can dump them (see Ps2DumpVif1Trace);
        // they are only ever written out with PS2X_VIF1_FAILDUMP, so skip the copy otherwise.
        // (boot.sh exports PS2X_VIF1_FAILDUMP= empty by default; Ps2DumpVif1Trace needs a directory.)
        static const bool keepRaw = []
        {
            const char *dir = std::getenv("PS2X_VIF1_FAILDUMP");
            return dir != nullptr && dir[0] != '\0';
        }();
        if (keepRaw)
        {
            auto &slot = g_vif1RawRing[g_vif1RawHead++ % kVif1RawRing];
            slot.assign(data, data + sizeBytes);
        }
    }

    uint32_t pos = 0;

    while (pos + 4 <= sizeBytes)
    {
        if (m_vif1PendingDirectQwc != 0u)
        {
            // Continuation of a DIRECT/DIRECTHL whose payload was split across DMA buffers.
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            if (availableQw == 0u)
                break;
            const uint32_t chunkQw = std::min<uint32_t>(m_vif1PendingDirectQwc, availableQw);
            feedVif1Path2(data + pos, chunkQw, m_vif1PendingDirectHl);
            pos += chunkQw * 16u;
            m_vif1PendingDirectQwc -= chunkQw;
            continue;
        }

        uint32_t cmd;
        memcpy(&cmd, data + pos, 4);
        pos += 4;

        uint8_t opcode = (cmd >> 24) & 0x7F;
        {   // statistics only (single writer: the EE thread); avoid a locked RMW per VIFcode
            auto &counter = g_vif1OpcodeCounts[opcode & 0x7F];
            counter.store(counter.load(std::memory_order_relaxed) + 1u, std::memory_order_relaxed);
        }
        uint16_t imm = cmd & 0xFFFF;
        uint8_t num = (cmd >> 16) & 0xFF;
        const bool irq = (cmd & 0x80000000u) != 0u;

        // Track most-recent command for VIFn_CODE emulation.
        vif1_regs.code = cmd;
        vif1_regs.num = num;
        if (irq)
            vif1_regs.stat |= (1u << 11); // INT

        {
            // Ring buffer of recent VIF1 commands + double-buffer state, dumped on the first bad XGKICK.
            Vif1TraceEntry &e = g_vif1Trace[g_vif1TraceHead++ % kVif1TraceSize];
            e.cmd = cmd;
            e.tops = vif1_regs.tops;
            e.base = vif1_regs.base;
            e.ofst = vif1_regs.ofst;
            e.dbf = (vif1_regs.stat >> 7) & 1u;
            e.firstWord = (pos + 4u <= sizeBytes) ? (uint32_t(data[pos]) | uint32_t(data[pos + 1]) << 8 | uint32_t(data[pos + 2]) << 16 | uint32_t(data[pos + 3]) << 24) : 0u;
        }

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif1_regs.cycle = imm;
            continue;
        }
        else if (opcode == VIF_OFFSET)
        {
            // VIF double-buffer setup. OFFSET clears DBF and resets TOPS to BASE.
            // Do not rewrite BASE from the previous TOPS value.
            vif1_regs.ofst = imm & 0x3FFu;
            vif1_regs.tops = vif1_regs.base & 0x3FFu;
            vif1_regs.stat &= ~(1u << 7); // clear DBF
            continue;
        }
        else if (opcode == VIF_BASE)
        {
            // BASE only updates the base register. TOPS changes on OFFSET/MSCAL.
            vif1_regs.base = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            // ITOP VIFcode writes pending ITOPS; VU XITOP observes it after MSCAL/MSCNT.
            vif1_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif1_regs.mode = imm & 3u;
            continue;
        }
        else if (opcode == VIF_MSKPATH3)
        {
            // VIF command docs: MSKPATH3 uses IMMEDIATE bit 15.
            const bool wasMasked = m_path3Masked;
            m_path3Masked = (imm & 0x8000u) != 0u;
            if (wasMasked && !m_path3Masked)
                flushMaskedPath3Packets();
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif1_regs.mark = imm;
            vif1_regs.stat |= (1u << 6); // MRK
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_MSCAL || opcode == VIF_MSCALF)
        {
            uint32_t startPC = (uint32_t)imm * 8u;

            const uint32_t runTop = vif1_regs.tops & 0x3FFu;
            const uint32_t runItop = vif1_regs.itops & 0x3FFu;
            vif1_regs.top = runTop;
            vif1_regs.itop = runItop;

            const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
            if (dbf)
                vif1_regs.tops = vif1_regs.base & 0x3FFu;
            else
                vif1_regs.tops = (vif1_regs.base + vif1_regs.ofst) & 0x3FFu;
            vif1_regs.stat ^= (1u << 7); // toggle DBF

            if (m_vu1MscalCallback)
                m_vu1MscalCallback(startPC, runTop, runItop);
            continue;
        }
        else if (opcode == VIF_MSCNT)
        {
            const uint32_t runTop = vif1_regs.tops & 0x3FFu;
            const uint32_t runItop = vif1_regs.itops & 0x3FFu;
            vif1_regs.top = runTop;
            vif1_regs.itop = runItop;

            const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
            if (dbf)
                vif1_regs.tops = vif1_regs.base & 0x3FFu;
            else
                vif1_regs.tops = (vif1_regs.base + vif1_regs.ofst) & 0x3FFu;
            vif1_regs.stat ^= (1u << 7); // toggle DBF

            if (m_vu1MscntCallback)
                m_vu1MscntCallback(runTop, runItop);
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4 > sizeBytes)
                break;
            uint32_t maskValue = 0;
            std::memcpy(&maskValue, data + pos, sizeof(maskValue));
            vif1_regs.mask = maskValue;
            pos += 4;
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16 > sizeBytes)
                break;
            std::memcpy(vif1_regs.row, data + pos, 16);
            pos += 16;
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16 > sizeBytes)
                break;
            std::memcpy(vif1_regs.col, data + pos, 16);
            pos += 16;
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            uint32_t destAddr = (uint32_t)imm * 8u;
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            if (m_vu1Code && destAddr < PS2_VU1_CODE_SIZE && mpgBytes > 0)
            {
                uint32_t copyBytes = mpgBytes;
                if (destAddr + copyBytes > PS2_VU1_CODE_SIZE)
                    copyBytes = PS2_VU1_CODE_SIZE - destAddr;
                if (pos + copyBytes <= sizeBytes)
                {
                    std::memcpy(m_vu1Code + destAddr, data + pos, copyBytes);
                    markVU1CodeModified();
                }
            }
            pos += mpgBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else if (opcode == VIF_DIRECT || opcode == VIF_DIRECTHL)
        {
            uint32_t qwCount = imm;
            if (qwCount == 0)
                qwCount = 65536;
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            const bool truncated = qwCount > availableQw;
            if (qwCount > availableQw)
                qwCount = availableQw;

            const bool directHl = (opcode == VIF_DIRECTHL);
            if (qwCount > 0)
                feedVif1Path2(data + pos, qwCount, directHl);
            pos += qwCount * 16;
            if (truncated)
            {
                // The rest of this DIRECT's payload arrives at the start of the next VIF1 buffer.
                m_vif1PendingDirectQwc = ((imm == 0u) ? 65536u : static_cast<uint32_t>(imm)) - qwCount;
                m_vif1PendingDirectHl = directHl;
                pos = sizeBytes;
                break;
            }
            continue;
        }
        else if ((opcode & 0x60) == 0x60)
        {
            uint8_t vn = (opcode >> 2) & 0x3;
            uint8_t vl = opcode & 0x3;
            const bool maskEnable = (opcode & 0x10u) != 0u;
            int components = vn + 1;
            int bitsPerComponent = 32;
            switch (vl)
            {
            case 0:
                bitsPerComponent = 32;
                break;
            case 1:
                bitsPerComponent = 16;
                break;
            case 2:
                bitsPerComponent = 8;
                break;
            case 3:
                bitsPerComponent = (vn == 3) ? 4 : 16;
                break;
            default:
                break;
            }
            int bitsPerVector = (vl == 3 && vn == 3) ? 16 : (components * bitsPerComponent);
            uint32_t bytesPerVector = (bitsPerVector + 7) / 8;
            // UNPACK semantics: NUM is 8-bit and NUM==0 means 256 vectors (writes).
            const uint32_t writeVectorCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);

            // STCYCL controls write cycles for UNPACK.
            uint32_t cl = vif1_regs.cycle & 0xFFu;
            uint32_t wl = (vif1_regs.cycle >> 8) & 0xFFu;
            if (cl == 0u)
                cl = 1u;
            if (wl == 0u)
                wl = 1u;

            uint32_t sourceVectorCount = writeVectorCount;
            if (cl < wl)
            {
                const uint32_t fullBlocks = writeVectorCount / wl;
                uint32_t remainder = writeVectorCount % wl;
                if (remainder > cl)
                    remainder = cl;
                sourceVectorCount = fullBlocks * cl + remainder;
            }

            uint32_t totalBytes = sourceVectorCount * bytesPerVector;
            totalBytes = (totalBytes + 3) & ~3u;

            uint32_t vuAddr = (uint32_t)imm & 0x3FFu;
            if ((imm & 0x8000u) != 0u)
                vuAddr = (vuAddr + (vif1_regs.tops & 0x3FFu)) & 0x3FFu;

            const bool zeroExtend = (imm & 0x4000u) != 0u;

            if (m_vu1Data && totalBytes > 0 && pos + totalBytes <= sizeBytes)
            {
                auto unpackGeneric = [&]()
                {
                const uint8_t *srcBase = data + pos;
                uint32_t srcIndex = 0u;
                for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
                {
                    const uint32_t cyclePos = writeIndex % wl;
                    const bool sourceAvailable = (cl >= wl) || (cyclePos < cl);

                    uint32_t destVec = 0;
                    if (cl >= wl)
                    {
                        destVec = (vuAddr + (writeIndex / wl) * cl + cyclePos) & 0x3FFu;
                    }
                    else
                    {
                        destVec = (vuAddr + writeIndex) & 0x3FFu;
                    }

                    uint32_t destOff = destVec * 16u;
                    if (destOff + 16u > PS2_VU1_DATA_SIZE)
                    {
                        if (sourceAvailable && srcIndex < sourceVectorCount)
                            ++srcIndex;
                        continue;
                    }

                    uint32_t lanes[4] = {0u, 0u, 0u, 0u};
                    std::memcpy(lanes, m_vu1Data + destOff, sizeof(lanes));
                    uint32_t decompressed[4] = {lanes[0], lanes[1], lanes[2], lanes[3]};
                    bool decoded = false;

                    const uint8_t *srcVec = nullptr;
                    if (sourceAvailable && srcIndex < sourceVectorCount)
                    {
                        srcVec = srcBase + srcIndex * bytesPerVector;
                        ++srcIndex;
                        decoded = true;
                    }

                    auto extend16 = [&](uint16_t raw) -> uint32_t
                    {
                        if (zeroExtend)
                            return static_cast<uint32_t>(raw);
                        return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(raw)));
                    };

                    auto extend8 = [&](uint8_t raw) -> uint32_t
                    {
                        if (zeroExtend)
                            return static_cast<uint32_t>(raw);
                        return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(raw)));
                    };

                    bool handledFormat = true;
                    if (!decoded)
                    {
                        handledFormat = false;
                    }
                    else if (vl == 0u)
                    {
                        if (components == 1)
                        {
                            uint32_t scalar = 0;
                            std::memcpy(&scalar, srcVec, sizeof(scalar));
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                uint32_t scalar = 0;
                                std::memcpy(&scalar, srcVec + c * 4u, sizeof(scalar));
                                decompressed[c] = scalar;
                            }
                        }
                    }
                    else if (vl == 1u)
                    {
                        if (components == 1)
                        {
                            uint16_t raw = 0;
                            std::memcpy(&raw, srcVec, sizeof(raw));
                            const uint32_t scalar = extend16(raw);
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                uint16_t raw = 0;
                                std::memcpy(&raw, srcVec + c * 2u, sizeof(raw));
                                decompressed[c] = extend16(raw);
                            }
                        }
                    }
                    else if (vl == 2u)
                    {
                        if (components == 1)
                        {
                            const uint32_t scalar = extend8(srcVec[0]);
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                decompressed[c] = extend8(srcVec[c]);
                            }
                        }
                    }
                    else if (vl == 3u && vn == 3u)
                    {
                        // V4-5: packed color-like format in a single 16-bit value.
                        uint16_t packed = 0;
                        std::memcpy(&packed, srcVec, sizeof(packed));
                        // RGBA5551 -> 8 bits per lane (hardware: R,G,B << 3, A << 7).
                        decompressed[0] = (packed & 0x1Fu) << 3u;
                        decompressed[1] = ((packed >> 5) & 0x1Fu) << 3u;
                        decompressed[2] = ((packed >> 10) & 0x1Fu) << 3u;
                        decompressed[3] = ((packed >> 15) & 0x01u) << 7u;
                    }
                    else
                    {
                        handledFormat = false;
                    }

                    // Unknown compressed format fallback: preserve legacy raw-copy behavior.
                    if (!handledFormat && decoded && !maskEnable && (vif1_regs.mode == 0u || vif1_regs.mode == 3u))
                    {
                        uint32_t copyBytes = (bytesPerVector < 16u) ? bytesPerVector : 16u;
                        std::memcpy(m_vu1Data + destOff, srcVec, copyBytes);
                        continue;
                    }

                    const bool canAdd = (vl != 3u || vn != 3u);
                    const uint32_t mode = vif1_regs.mode & 3u;
                    const uint32_t colIdx = (cyclePos > 3u) ? 3u : cyclePos;
                    const uint32_t maskCycle = (cyclePos > 3u) ? 3u : cyclePos;

                    for (uint32_t field = 0u; field < 4u; ++field)
                    {
                        uint32_t maskSpec = 0u;
                        if (maskEnable)
                        {
                            const uint32_t shift = ((maskCycle * 4u) + field) * 2u;
                            maskSpec = (vif1_regs.mask >> shift) & 0x3u;
                        }

                        // In fill-write cycles with suspended source reads, treat raw-data selections as row-fill.
                        if (!decoded && maskSpec == 0u)
                            maskSpec = 1u;

                        uint32_t writeVal = lanes[field];
                        if (maskSpec == 0u)
                        {
                            if (handledFormat)
                            {
                                writeVal = decompressed[field];
                                if (canAdd && (mode == 1u || mode == 2u))
                                {
                                    writeVal = writeVal + vif1_regs.row[field];
                                    if (mode == 2u)
                                        vif1_regs.row[field] = writeVal;
                                }
                            }
                        }
                        else if (maskSpec == 1u)
                        {
                            writeVal = vif1_regs.row[field];
                        }
                        else if (maskSpec == 2u)
                        {
                            writeVal = vif1_regs.col[colIdx];
                        }
                        else
                        {
                            continue; // write-protect
                        }

                        lanes[field] = writeVal;
                    }

                    if (destVec >= 0x3F4u && destVec <= 0x3F5u)
                    {
                        // (print budget shared with the fast path: g_vu1WatchPrints)
                        if (g_vu1WatchPrints++ < 16)
                            std::fprintf(stderr, "[vu1-watch] VIF unpack qw=0x%x op=0x%x imm=0x%x num=%u tops=0x%x words=%08x %08x %08x %08x\n",
                                         destVec, opcode, imm, num, vif1_regs.tops, lanes[0], lanes[1], lanes[2], lanes[3]);
                    }
                    std::memcpy(m_vu1Data + destOff, lanes, sizeof(lanes));
                }
                };

                // Fast path (bit-identical to unpackGeneric): no write mask, CL >= WL (every write cycle
                // has source data, so no row/col fill), and a decodable format. PS2X_VIF_VERIFY=1 runs
                // both implementations on every such UNPACK and reports any difference.
                const bool fastOk = !maskEnable && cl >= wl && (vl != 3u || vn == 3u) &&
                                    !vif1UnpackTouchesWatch(vuAddr, writeVectorCount, cl, wl);
                if (!fastOk)
                {
                    unpackGeneric();
                }
                else if (!vif1VerifyEnabled())
                {
                    vif1UnpackFast(m_vu1Data, vif1_regs.row, data + pos, vuAddr, writeVectorCount, cl, wl,
                                   vn, vl, vif1_regs.mode & 3u, zeroExtend);
                }
                else
                {
                    alignas(16) static uint8_t before[PS2_VU1_DATA_SIZE], fast[PS2_VU1_DATA_SIZE];
                    uint32_t rowBefore[4], rowFast[4];
                    std::memcpy(before, m_vu1Data, PS2_VU1_DATA_SIZE);
                    std::memcpy(rowBefore, vif1_regs.row, sizeof(rowBefore));
                    vif1UnpackFast(m_vu1Data, vif1_regs.row, data + pos, vuAddr, writeVectorCount, cl, wl,
                                   vn, vl, vif1_regs.mode & 3u, zeroExtend);
                    std::memcpy(fast, m_vu1Data, PS2_VU1_DATA_SIZE);
                    std::memcpy(rowFast, vif1_regs.row, sizeof(rowFast));
                    std::memcpy(m_vu1Data, before, PS2_VU1_DATA_SIZE);
                    std::memcpy(vif1_regs.row, rowBefore, sizeof(rowBefore));
                    unpackGeneric();
                    vif1VerifyReport(opcode, imm, num, vif1_regs.cycle, vif1_regs.mode,
                                     std::memcmp(fast, m_vu1Data, PS2_VU1_DATA_SIZE) == 0 &&
                                         std::memcmp(rowFast, vif1_regs.row, sizeof(rowFast)) == 0);
                }
            }
            pos += totalBytes;

            if (pos > sizeBytes)
                break;
            continue;
        }
        else
        {
            continue;
        }
    }
}
