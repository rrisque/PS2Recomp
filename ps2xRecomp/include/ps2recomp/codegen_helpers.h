#ifndef PS2RECOMP_CODEGEN_HELPERS_H
#define PS2RECOMP_CODEGEN_HELPERS_H

#include <cmath>
#include <cstdint>
#include <string>
#include <fmt/format.h>

namespace ps2recomp::codegen
{
    inline std::string formatFloatLiteral(float value)
    {
        if (!std::isfinite(value))
        {
            return (value < 0.0f) ? "-INFINITY" : "INFINITY";
        }

        std::string literal = fmt::format("{:.9g}", value);
        if (literal.find_first_of(".eE") == std::string::npos)
        {
            literal += ".0";
        }
        literal += 'f';
        return literal;
    }

    inline std::string vuMaskExpr(uint8_t dest_mask)
    {
        return fmt::format("_mm_castsi128_ps(_mm_set_epi32({}, {}, {}, {}))",
                           (dest_mask & 0x1) ? -1 : 0,
                           (dest_mask & 0x2) ? -1 : 0,
                           (dest_mask & 0x4) ? -1 : 0,
                           (dest_mask & 0x8) ? -1 : 0);
    }

    // VU0 register vf<reg> broadcast of one component (0=x .. 3=w).
    inline std::string vuBroadcast(uint8_t reg, uint8_t field)
    {
        field &= 3;
        return fmt::format("_mm_shuffle_ps(ctx->vu0_vf[{0}], ctx->vu0_vf[{0}], _MM_SHUFFLE({1},{1},{1},{1}))", reg, field);
    }

    // One component of vf<reg> as a float.
    inline std::string vuLane(uint8_t reg, uint8_t field)
    {
        return fmt::format("_mm_cvtss_f32(_mm_shuffle_ps(ctx->vu0_vf[{0}], ctx->vu0_vf[{0}], _MM_SHUFFLE(0,0,0,{1})))", reg, field & 3);
    }

    // Masked write of the __m128 expression `res` to a VU0 destination expression (vf register or ACC).
    // dest_mask uses the instruction encoding: bit3=x, bit2=y, bit1=z, bit0=w.
    inline std::string vuMaskedStore(const std::string &dst, uint8_t dest_mask, const std::string &res)
    {
        dest_mask &= 0xF;
        if (dest_mask == 0)
            return "/* empty dest mask */";
        if (dest_mask == 0xF)
            return fmt::format("{} = {};", dst, res);
        return fmt::format("{0} = _mm_blendv_ps({0}, {1}, {2});", dst, res, vuMaskExpr(dest_mask));
    }

    // Masked write to vf<vfd>. vf0 is hardwired to (0,0,0,1): writes to it are discarded.
    inline std::string vuStoreVF(uint8_t vfd, uint8_t dest_mask, const std::string &res)
    {
        if (vfd == 0)
            return "/* write to vf0 discarded */";
        return vuMaskedStore(fmt::format("ctx->vu0_vf[{}]", vfd), dest_mask, res);
    }

    // Write to vi<vid> (16-bit). vi0 is hardwired to 0: writes to it are discarded.
    inline std::string vuStoreVI(uint8_t vid, const std::string &value)
    {
        vid &= 0xF;
        if (vid == 0)
            return "/* write to vi0 discarded */";
        return fmt::format("ctx->vi[{}] = static_cast<uint16_t>({});", vid, value);
    }

    // VU0 data memory (4KB, mapped at 0x11004000 on the EE) address of quadword vi<reg>.
    inline std::string vuMemAddr(uint8_t reg)
    {
        return fmt::format("(0x11004000u + ((static_cast<uint32_t>(ctx->vi[{}]) << 4) & 0xFF0u))", reg & 0xF);
    }
}

#endif // PS2RECOMP_CODEGEN_HELPERS_H
