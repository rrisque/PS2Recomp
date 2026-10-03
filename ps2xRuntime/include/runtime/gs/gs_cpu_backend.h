#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_texture_page_cache.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <thread>
#include <mutex>
#include <vector>

class GSCpuBackend final : public GSRasterBackend
{
public:
    GSCpuBackend();
    ~GSCpuBackend() override;
    GSCpuBackend(const GSCpuBackend &) = delete;
    GSCpuBackend &operator=(const GSCpuBackend &) = delete;

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

    // Capture/replay support (see gs_capture.h).
    std::vector<uint8_t> SerializeState() const;
    bool RestoreState(const uint8_t *data, size_t size);

private:
    // Command execution (caller holds m_mutex; runs on the GS worker thread when async).
    void ExecSubmit(const GSPrimitiveBatch &batch);
    void ExecLoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    void ExecBeginTransfer(const GSTransferCommand &command);
    void ExecUploadImage(const uint8_t *data, uint32_t sizeBytes);
    void ExecTextureFlush();
    void ExecReset();
    bool ExecClearFramebuffer(const GSContext &context, uint32_t rgba);
    uint32_t ExecConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes);
    void ExecWriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);

    // Asynchronous execution: mutating calls are queued in order and executed by one worker
    // thread; calls that observe VRAM wait for the queue to drain first (WaitIdle).
    struct AsyncQueue;
    bool AsyncActive() const { return m_asyncEnabled; }
    void Enqueue(uint32_t type, const void *a, uint32_t aLen, const void *b = nullptr, uint32_t bLen = 0);
    void WaitIdle() const;
    void WorkerMain();
    void ExecuteCommand(uint32_t type, const uint8_t *payload, uint32_t len);
    void StopWorker();

    void ResetUnlocked();
    bool CaptureTick();
    std::vector<uint8_t> SerializeStateUnlocked() const;
    void LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    uint32_t ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const;
    uint32_t ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y);
    void WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);

    void DrawPrimitive(const GSPrimitiveBatch &batch);
    void DrawSprite(const GSPrimitiveBatch &batch);
    void DrawTriangle(const GSPrimitiveBatch &batch);
    void DrawLine(const GSPrimitiveBatch &batch);
    void DrawPoint(const GSPrimitiveBatch &batch);
    void WritePixel(const GSDrawState &state, int x, int y, uint32_t z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog);
    // u/v: level-0 texel coordinates in 16.16 fixed point (pixel-centre sampling, no half-texel bias);
    // q: interpolated Q, used for the TEX1 LOD when LCM=0 and FST=0.
    uint32_t SampleTexture(const GSDrawState &state, int32_t u, int32_t v, float q);
    uint32_t FetchTexel(const GSDrawState &state, uint32_t tbp, uint32_t tbw, int u, int v);
    uint32_t LookupCLUT(const GSDrawState &state, uint8_t index, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm);

    // Optimised rasterizer (bit-identical to the reference paths above, which remain the
    // fallback for state combinations the fast path does not cover).
    friend struct GSFastRaster;
    struct ParallelState;
    ParallelState *ParallelFor();
    void FlushParallel();
    bool DeferClutLoad(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    bool DeferUpload() const;
    void DeferTextureFlush();
    bool DrawTriangleFast(const GSPrimitiveBatch &batch);
    bool DrawSpriteFast(const GSPrimitiveBatch &batch);
    const uint32_t *GetPalette(const GSDrawState &state);

    void PerformLocalToLocalTransfer();
    void PerformLocalToHostTransfer();
    PresentationFrame PresentFromLocalMemory(const GSPresentationRequest &request);
    bool CopyFrameToHostRgba(const GSFrameReg &frame,
                             uint32_t width,
                             uint32_t height,
                             std::vector<uint8_t> &outPixels,
                             bool preserveAlpha,
                             bool useLocalMemoryLayout,
                             bool frameBaseIsPages,
                             uint32_t sourceOriginX,
                             uint32_t sourceOriginY) const;

    using WriteVramFunc = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using ReadVramFunc = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);

    static constexpr size_t kPsmHandlerCount = 1u << 6u;
    mutable std::mutex m_mutex;
    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;
    std::array<ReadVramFunc, kPsmHandlerCount> m_readVramFuncs{};
    std::array<WriteVramFunc, kPsmHandlerCount> m_writeVramFuncs{};
    std::array<uint16_t, 512> m_clut{};
    std::array<uint32_t, 2> m_clutCbp{};
    GSMem::TexturePageCache m_texturePageCache;
    uint64_t m_clutVersion = 1;
    uint64_t m_paletteVersion = 0;
    uint64_t m_paletteKey = ~0ull;
    alignas(64) std::array<uint32_t, 256> m_palette{};

    bool m_asyncEnabled = false;
    std::unique_ptr<AsyncQueue> m_async;
    std::unique_ptr<ParallelState> m_par;
    bool m_parDisabled = false;

    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;
};
