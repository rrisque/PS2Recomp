#include "ps2recomp/code_generator.h"
#include "ps2recomp/codegen_helpers.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"
#include <fmt/format.h>
#include <sstream>
#include <cmath>

// VU0 macro-mode (COP2) instruction translation.
// Semantics follow the PCSX2 interpreter (pcsx2/VUops.cpp): float operands/results are normalised by the
// runtime helpers (no Inf/NaN/denormal), vf0 and vi0 are read-only, vi registers are 16-bit and indexed
// with the low 4 bits of the field, VU0 data memory is 4KB at 0x11004000.
// MAC/status flag side effects of the FMAC ops are not modelled.

namespace ps2recomp
{
    namespace
    {
        using codegen::vuBroadcast;
        using codegen::vuLane;
        using codegen::vuMaskedStore;
        using codegen::vuMemAddr;
        using codegen::vuStoreVF;
        using codegen::vuStoreVI;

        // fd = expr (masked); a discarded vf0 write emits nothing (no flags are modelled).
        std::string emitVF(uint8_t vfd, uint8_t dest_mask, const std::string &expr)
        {
            if (vfd == 0)
                return "/* write to vf0 discarded */";
            return fmt::format("{{ __m128 res = {}; {} }}", expr, vuStoreVF(vfd, dest_mask, "res"));
        }

        std::string emitACC(uint8_t dest_mask, const std::string &expr)
        {
            return fmt::format("{{ __m128 res = {}; {} }}", expr, vuMaskedStore("ctx->vu0_acc", dest_mask, "res"));
        }

        std::string vf(uint8_t reg) { return fmt::format("ctx->vu0_vf[{}]", reg); }
        const char *kQ = "_mm_set1_ps(ctx->vu0_q)";
        const char *kI = "_mm_set1_ps(ctx->vu0_i)";
    }

    // ---- ADD / SUB / MUL ---------------------------------------------------------------------------------

    std::string CodeGenerator::translateVU_VADD_Field(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VADD({}, {})", vf(inst.rd), vuBroadcast(inst.rt, inst.function & 3)));
    }

    std::string CodeGenerator::translateVU_VSUB_Field(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VSUB({}, {})", vf(inst.rd), vuBroadcast(inst.rt, inst.function & 3)));
    }

    std::string CodeGenerator::translateVU_VMUL_Field(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMUL({}, {})", vf(inst.rd), vuBroadcast(inst.rt, inst.function & 3)));
    }

    std::string CodeGenerator::translateVU_VADD(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VADD({}, {})", vf(inst.rd), vf(inst.rt)));
    }

    std::string CodeGenerator::translateVU_VSUB(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VSUB({}, {})", vf(inst.rd), vf(inst.rt)));
    }

    std::string CodeGenerator::translateVU_VMUL(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMUL({}, {})", vf(inst.rd), vf(inst.rt)));
    }

    std::string CodeGenerator::translateVU_VADDq(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VADD({}, {})", vf(inst.rd), kQ));
    }

    std::string CodeGenerator::translateVU_VADDi(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VADD({}, {})", vf(inst.rd), kI));
    }

    std::string CodeGenerator::translateVU_VSUBq(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VSUB({}, {})", vf(inst.rd), kQ));
    }

    std::string CodeGenerator::translateVU_VSUBi(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VSUB({}, {})", vf(inst.rd), kI));
    }

    std::string CodeGenerator::translateVU_VMULq(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMUL({}, {})", vf(inst.rd), kQ));
    }

    std::string CodeGenerator::translateVU_VMULi(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMUL({}, {})", vf(inst.rd), kI));
    }

    // ---- MADD / MSUB -------------------------------------------------------------------------------------

    std::string CodeGenerator::translateVU_VMADD_Field(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMADD(ctx->vu0_acc, {}, {})", vf(inst.rd), vuBroadcast(inst.rt, inst.function & 3)));
    }

    std::string CodeGenerator::translateVU_VMSUB_Field(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMSUB(ctx->vu0_acc, {}, {})", vf(inst.rd), vuBroadcast(inst.rt, inst.function & 3)));
    }

    std::string CodeGenerator::translateVU_VMADD(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMADD(ctx->vu0_acc, {}, {})", vf(inst.rd), vf(inst.rt)));
    }

    std::string CodeGenerator::translateVU_VMADDq(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMADD(ctx->vu0_acc, {}, {})", vf(inst.rd), kQ));
    }

    std::string CodeGenerator::translateVU_VMADDi(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMADD(ctx->vu0_acc, {}, {})", vf(inst.rd), kI));
    }

    std::string CodeGenerator::translateVU_VMSUB(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMSUB(ctx->vu0_acc, {}, {})", vf(inst.rd), vf(inst.rt)));
    }

    std::string CodeGenerator::translateVU_VMSUBq(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMSUB(ctx->vu0_acc, {}, {})", vf(inst.rd), kQ));
    }

    std::string CodeGenerator::translateVU_VMSUBi(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMSUB(ctx->vu0_acc, {}, {})", vf(inst.rd), kI));
    }

    // VOPMSUB.xyz fd, fs, ft: fd.xyz = ACC.xyz - fs.yzx * ft.zxy (always xyz; w untouched).
    std::string CodeGenerator::translateVU_VOPMSUB(const Instruction &inst)
    {
        return emitVF(inst.sa, 0xE,
                      fmt::format("PS2_VMSUB(ctx->vu0_acc, _mm_shuffle_ps({0}, {0}, _MM_SHUFFLE(3,0,2,1)), _mm_shuffle_ps({1}, {1}, _MM_SHUFFLE(3,1,0,2)))",
                                  vf(inst.rd), vf(inst.rt)));
    }

    // ---- MAX / MINI (integer compare of the bit patterns, PCSX2 fp_max/fp_min) ---------------------------

    std::string CodeGenerator::translateVU_VMAX_Field(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMAX({}, {})", vf(inst.rd), vuBroadcast(inst.rt, inst.function & 3)));
    }

    std::string CodeGenerator::translateVU_VMINI_Field(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMINI({}, {})", vf(inst.rd), vuBroadcast(inst.rt, inst.function & 3)));
    }

    std::string CodeGenerator::translateVU_VMAX(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMAX({}, {})", vf(inst.rd), vf(inst.rt)));
    }

    std::string CodeGenerator::translateVU_VMINI(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMINI({}, {})", vf(inst.rd), vf(inst.rt)));
    }

    std::string CodeGenerator::translateVU_VMAXi(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMAX({}, {})", vf(inst.rd), kI));
    }

    std::string CodeGenerator::translateVU_VMINIi(const Instruction &inst)
    {
        return emitVF(inst.sa, inst.vectorInfo.vectorField, fmt::format("PS2_VMINI({}, {})", vf(inst.rd), kI));
    }

    // ---- Accumulator forms -------------------------------------------------------------------------------

    std::string CodeGenerator::translateVU_VADDA_Field(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VADD({}, {})", vf(inst.rd), vuBroadcast(inst.rt, inst.function & 3)));
    }

    std::string CodeGenerator::translateVU_VSUBA_Field(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VSUB({}, {})", vf(inst.rd), vuBroadcast(inst.rt, inst.function & 3)));
    }

    std::string CodeGenerator::translateVU_VMADDA_Field(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VMADD(ctx->vu0_acc, {}, {})", vf(inst.rd), vuBroadcast(inst.rt, inst.function & 3)));
    }

    std::string CodeGenerator::translateVU_VMSUBA_Field(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VMSUB(ctx->vu0_acc, {}, {})", vf(inst.rd), vuBroadcast(inst.rt, inst.function & 3)));
    }

    std::string CodeGenerator::translateVU_VMULA_Field(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VMUL({}, {})", vf(inst.rd), vuBroadcast(inst.rt, inst.function & 3)));
    }

    std::string CodeGenerator::translateVU_VADDA(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VADD({}, {})", vf(inst.rd), vf(inst.rt)));
    }

    std::string CodeGenerator::translateVU_VADDAq(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VADD({}, {})", vf(inst.rd), kQ));
    }

    std::string CodeGenerator::translateVU_VADDAi(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VADD({}, {})", vf(inst.rd), kI));
    }

    std::string CodeGenerator::translateVU_VSUBA(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VSUB({}, {})", vf(inst.rd), vf(inst.rt)));
    }

    std::string CodeGenerator::translateVU_VSUBAq(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VSUB({}, {})", vf(inst.rd), kQ));
    }

    std::string CodeGenerator::translateVU_VSUBAi(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VSUB({}, {})", vf(inst.rd), kI));
    }

    std::string CodeGenerator::translateVU_VMADDA(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VMADD(ctx->vu0_acc, {}, {})", vf(inst.rd), vf(inst.rt)));
    }

    std::string CodeGenerator::translateVU_VMADDAq(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VMADD(ctx->vu0_acc, {}, {})", vf(inst.rd), kQ));
    }

    std::string CodeGenerator::translateVU_VMADDAi(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VMADD(ctx->vu0_acc, {}, {})", vf(inst.rd), kI));
    }

    std::string CodeGenerator::translateVU_VMSUBA(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VMSUB(ctx->vu0_acc, {}, {})", vf(inst.rd), vf(inst.rt)));
    }

    std::string CodeGenerator::translateVU_VMSUBAq(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VMSUB(ctx->vu0_acc, {}, {})", vf(inst.rd), kQ));
    }

    std::string CodeGenerator::translateVU_VMSUBAi(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VMSUB(ctx->vu0_acc, {}, {})", vf(inst.rd), kI));
    }

    std::string CodeGenerator::translateVU_VMULA(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VMUL({}, {})", vf(inst.rd), vf(inst.rt)));
    }

    std::string CodeGenerator::translateVU_VMULAq(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VMUL({}, {})", vf(inst.rd), kQ));
    }

    std::string CodeGenerator::translateVU_VMULAi(const Instruction &inst)
    {
        return emitACC(inst.vectorInfo.vectorField, fmt::format("PS2_VMUL({}, {})", vf(inst.rd), kI));
    }

    // VOPMULA.xyz ACC, fs, ft: ACC.xyz = fs.yzx * ft.zxy (always xyz).
    std::string CodeGenerator::translateVU_VOPMULA(const Instruction &inst)
    {
        return emitACC(0xE, fmt::format("PS2_VMUL(_mm_shuffle_ps({0}, {0}, _MM_SHUFFLE(3,0,2,1)), _mm_shuffle_ps({1}, {1}, _MM_SHUFFLE(3,1,0,2)))",
                                        vf(inst.rd), vf(inst.rt)));
    }

    // ---- Conversions -------------------------------------------------------------------------------------

    std::string CodeGenerator::translateVU_VITOF(const Instruction &inst, int shift)
    {
        const float scale = 1.0f / static_cast<float>(1 << shift);
        return emitVF(inst.rt, inst.vectorInfo.vectorField, fmt::format("Ps2VItof({}, {})", vf(inst.rd), codegen::formatFloatLiteral(scale)));
    }

    std::string CodeGenerator::translateVU_VFTOI(const Instruction &inst, int shift)
    {
        const float scale = static_cast<float>(1 << shift);
        return emitVF(inst.rt, inst.vectorInfo.vectorField, fmt::format("Ps2VFtoi({}, {})", vf(inst.rd), codegen::formatFloatLiteral(scale)));
    }

    // ---- FDIV unit (Q) -----------------------------------------------------------------------------------

    std::string CodeGenerator::translateVU_VDIV(const Instruction &inst)
    {
        return fmt::format("ctx->vu0_q = Ps2FDivAt(ctx->pc, {}, {});",
                           vuLane(inst.rd, inst.vectorInfo.fsf), vuLane(inst.rt, inst.vectorInfo.ftf));
    }

    std::string CodeGenerator::translateVU_VSQRT(const Instruction &inst)
    {
        return fmt::format("ctx->vu0_q = Ps2VSqrt({});", vuLane(inst.rt, inst.vectorInfo.ftf));
    }

    std::string CodeGenerator::translateVU_VRSQRT(const Instruction &inst)
    {
        // VRSQRT Q, fs.fsf, ft.ftf: Q = fs / sqrt(|ft|)
        return fmt::format("ctx->vu0_q = Ps2VRsqrtAt(ctx->pc, {}, {});",
                           vuLane(inst.rd, inst.vectorInfo.fsf), vuLane(inst.rt, inst.vectorInfo.ftf));
    }

    // ---- Integer unit ------------------------------------------------------------------------------------

    // VMTIR it, fs.fsf: it = low 16 bits of fs.fsf
    std::string CodeGenerator::translateVU_VMTIR(const Instruction &inst)
    {
        return fmt::format("{{ {} }}", vuStoreVI(inst.rt, fmt::format("Ps2FloatBits({}) & 0xFFFFu", vuLane(inst.rd, inst.vectorInfo.fsf))));
    }

    // VMFIR.dest ft, is: ft = sign-extended is (as integer bits)
    std::string CodeGenerator::translateVU_VMFIR(const Instruction &inst)
    {
        return emitVF(inst.rt, inst.vectorInfo.vectorField,
                      fmt::format("_mm_castsi128_ps(_mm_set1_epi32(static_cast<int32_t>(static_cast<int16_t>(ctx->vi[{}]))))", inst.rd & 0xF));
    }

    // VILWR.dest it, (is): it = 16-bit word at component <dest> of VU0 mem quadword is (last selected wins)
    std::string CodeGenerator::translateVU_VILWR(const Instruction &inst)
    {
        const uint8_t mask = inst.vectorInfo.vectorField & 0xF;
        int lane = -1;
        for (int l = 0; l < 4; ++l)
        {
            if (mask & (0x8 >> l))
                lane = l;
        }
        if (lane < 0 || (inst.rt & 0xF) == 0)
            return "/* VILWR: no effect */";
        return fmt::format("{{ uint32_t addr = {} + {}u; {} }}", vuMemAddr(inst.rd), lane * 4,
                           vuStoreVI(inst.rt, "READ16(addr)"));
    }

    // VISWR.dest it, (is): each selected 32-bit component of VU0 mem quadword is = zero-extended it
    std::string CodeGenerator::translateVU_VISWR(const Instruction &inst)
    {
        const uint8_t mask = inst.vectorInfo.vectorField & 0xF;
        std::string body = fmt::format("{{ uint32_t addr = {}; uint32_t val = static_cast<uint32_t>(ctx->vi[{}]); ", vuMemAddr(inst.rd), inst.rt & 0xF);
        for (int l = 0; l < 4; ++l)
        {
            if (mask & (0x8 >> l))
                body += fmt::format("WRITE32(addr + {}u, val); ", l * 4);
        }
        return body + "}";
    }

    std::string CodeGenerator::translateVU_VIADD(const Instruction &inst)
    {
        return vuStoreVI(inst.sa, fmt::format("ctx->vi[{}] + ctx->vi[{}]", inst.rd & 0xF, inst.rt & 0xF)); // vid, vis, vit
    }

    std::string CodeGenerator::translateVU_VISUB(const Instruction &inst)
    {
        return vuStoreVI(inst.sa, fmt::format("ctx->vi[{}] - ctx->vi[{}]", inst.rd & 0xF, inst.rt & 0xF));
    }

    std::string CodeGenerator::translateVU_VIADDI(const Instruction &inst)
    {
        const int32_t imm5 = (inst.sa & 0x10) ? static_cast<int32_t>(inst.sa | ~0x1F) : static_cast<int32_t>(inst.sa);
        return vuStoreVI(inst.rt, fmt::format("ctx->vi[{}] + ({})", inst.rd & 0xF, imm5)); // vit, vis, imm5
    }

    std::string CodeGenerator::translateVU_VIAND(const Instruction &inst)
    {
        return vuStoreVI(inst.sa, fmt::format("ctx->vi[{}] & ctx->vi[{}]", inst.rd & 0xF, inst.rt & 0xF));
    }

    std::string CodeGenerator::translateVU_VIOR(const Instruction &inst)
    {
        return vuStoreVI(inst.sa, fmt::format("ctx->vi[{}] | ctx->vi[{}]", inst.rd & 0xF, inst.rt & 0xF));
    }

    // ---- VU0 data memory load/store with post-increment / pre-decrement ----------------------------------

    // VLQI.dest ft, (is++)
    std::string CodeGenerator::translateVU_VLQI(const Instruction &inst)
    {
        const uint8_t is = inst.rd & 0xF;
        std::string load = (inst.rt == 0) ? std::string()
                                          : fmt::format("{{ __m128 res = _mm_castsi128_ps(READ128({})); {} }} ", vuMemAddr(is), vuStoreVF(inst.rt, inst.vectorInfo.vectorField, "res"));
        std::string inc = (is == 0) ? std::string() : fmt::format("ctx->vi[{0}] = static_cast<uint16_t>(ctx->vi[{0}] + 1);", is);
        return fmt::format("{{ {}{} }}", load, inc);
    }

    // VSQI.dest fs, (it++)
    std::string CodeGenerator::translateVU_VSQI(const Instruction &inst)
    {
        const uint8_t it = inst.rt & 0xF;
        std::string inc = (it == 0) ? std::string() : fmt::format("ctx->vi[{0}] = static_cast<uint16_t>(ctx->vi[{0}] + 1);", it);
        return fmt::format("{{ uint32_t addr = {}; __m128 res = {}; "
                           "res = _mm_blendv_ps(_mm_castsi128_ps(READ128(addr)), res, {}); "
                           "WRITE128(addr, _mm_castps_si128(res)); {} }}",
                           vuMemAddr(it), vf(inst.rd), codegen::vuMaskExpr(inst.vectorInfo.vectorField), inc);
    }

    // VLQD.dest ft, (--is)
    std::string CodeGenerator::translateVU_VLQD(const Instruction &inst)
    {
        const uint8_t is = inst.rd & 0xF;
        std::string dec = (is == 0) ? std::string() : fmt::format("ctx->vi[{0}] = static_cast<uint16_t>(ctx->vi[{0}] - 1); ", is);
        std::string load = (inst.rt == 0) ? std::string()
                                          : fmt::format("{{ __m128 res = _mm_castsi128_ps(READ128({})); {} }}", vuMemAddr(is), vuStoreVF(inst.rt, inst.vectorInfo.vectorField, "res"));
        return fmt::format("{{ {}{} }}", dec, load);
    }

    // VSQD.dest fs, (--it)
    std::string CodeGenerator::translateVU_VSQD(const Instruction &inst)
    {
        const uint8_t it = inst.rt & 0xF;
        std::string dec = (it == 0) ? std::string() : fmt::format("ctx->vi[{0}] = static_cast<uint16_t>(ctx->vi[{0}] - 1); ", it);
        return fmt::format("{{ {}uint32_t addr = {}; __m128 res = {}; "
                           "res = _mm_blendv_ps(_mm_castsi128_ps(READ128(addr)), res, {}); "
                           "WRITE128(addr, _mm_castps_si128(res)); }}",
                           dec, vuMemAddr(it), vf(inst.rd), codegen::vuMaskExpr(inst.vectorInfo.vectorField));
    }

    // ---- Micro-program calls -----------------------------------------------------------------------------

    std::string CodeGenerator::translateVU_VCALLMS(const Instruction &inst)
    {
        // VCALLMS imm15: start the VU0 micro program at instruction imm15 (VU0 micro memory: 512 instructions).
        uint16_t instr_index = static_cast<uint16_t>((inst.raw >> 6) & 0x1FF);
        uint32_t target_byte_addr = static_cast<uint32_t>(instr_index) << 3;

        return fmt::format(
            "{{ "
            "    ctx->vu0_tpc = 0x{:X}; "
            "    runtime->executeVU0Microprogram(rdram, ctx, 0x{:X}); "
            "}}",
            target_byte_addr, target_byte_addr);
    }

    std::string CodeGenerator::translateVU_VCALLMSR(const Instruction &inst)
    {
        // VCALLMSR vi27: start the VU0 micro program at the instruction index held in CMSAR0.
        (void)inst;
        return "{ "
               "    uint32_t target_byte_addr = (ctx->vu0_cmsar0 & 0x1FFu) << 3; "
               "    ctx->vu0_tpc = target_byte_addr; "
               "    runtime->vu0StartMicroProgram(rdram, ctx, target_byte_addr); "
               "}";
    }

    // ---- Random unit (R = 23-bit LFSR | 0x3F800000, kept splatted in ctx->vu0_r) ------------------------

    std::string CodeGenerator::translateVU_VRNEXT(const Instruction &inst)
    {
        if (inst.rt == 0)
            return "/* VRNEXT to vf0: no effect */";
        return fmt::format("{{ ctx->vu0_r = Ps2VuRNext(ctx->vu0_r); __m128 res = ctx->vu0_r; {} }}",
                           vuStoreVF(inst.rt, inst.vectorInfo.vectorField, "res"));
    }

    std::string CodeGenerator::translateVU_VRGET(const Instruction &inst)
    {
        return emitVF(inst.rt, inst.vectorInfo.vectorField, "ctx->vu0_r");
    }

    std::string CodeGenerator::translateVU_VRINIT(const Instruction &inst)
    {
        return fmt::format("ctx->vu0_r = Ps2VuRSet(Ps2FloatBits({}));", vuLane(inst.rd, inst.vectorInfo.fsf));
    }

    std::string CodeGenerator::translateVU_VRXOR(const Instruction &inst)
    {
        return fmt::format("ctx->vu0_r = Ps2VuRSet(Ps2VuRGet(ctx->vu0_r) ^ Ps2FloatBits({}));", vuLane(inst.rd, inst.vectorInfo.fsf));
    }

}
