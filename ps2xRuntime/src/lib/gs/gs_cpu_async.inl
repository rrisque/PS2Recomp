// Asynchronous command execution for GSCpuBackend (included from gs_cpu_backend.cpp).
//
// All state-mutating backend calls are appended to a single-consumer ring buffer and executed,
// strictly in submission order, by one GS worker thread. VRAM results are therefore identical
// to synchronous execution. Calls that observe backend state (VRAM reads/snapshots, local->host
// data, transfer state, presentation) first wait until every previously queued command has
// executed. PS2X_GS_ASYNC=0 disables the worker (commands then run on the calling thread).

struct GSCpuBackend::AsyncQueue
{
    static constexpr uint64_t kCapacity = 8ull * 1024ull * 1024ull; // bytes, multiple of 8
    static constexpr uint32_t kWrap = 0xFFFFFFFFu;

    std::unique_ptr<uint8_t[]> buffer{new uint8_t[kCapacity]};

    // Monotonic byte positions: head = bytes published by producers, tail = bytes executed.
    alignas(64) std::atomic<uint64_t> head{0};
    alignas(64) std::atomic<uint64_t> tail{0};

    alignas(64) std::mutex produceMutex; // serialises producers
    std::mutex waitMutex;
    std::condition_variable workCv;  // worker sleeps here
    std::condition_variable doneCv;  // producers waiting for space / idle sleep here
    std::atomic<bool> workerSleeping{false};
    std::atomic<int> doneWaiters{0};
    std::atomic<bool> stop{false};
    std::thread worker;
    bool started = false;
};

void GSCpuBackend::StopWorker()
{
    if (!m_async || !m_async->started)
        return;
    WaitIdle();
    {
        std::lock_guard<std::mutex> lk(m_async->waitMutex);
        m_async->stop.store(true);
    }
    m_async->workCv.notify_all();
    if (m_async->worker.joinable())
        m_async->worker.join();
    m_async->started = false;
}

void GSCpuBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    if (vram && vramSize < GSMem::MEMORY_SIZE)
        throw std::invalid_argument("GS CPU backend requires at least 4 MiB of VRAM");

    WaitIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    m_vram = vram;
    m_vramSize = vramSize;
    ResetUnlocked();
    // PS2X_GS_THREAD_VERIFY compares VRAM directly (bypassing the backend API) at GS fences, so
    // every command must have executed when the call that submitted it returns.
    static const bool verifyMode = std::getenv("PS2X_GS_THREAD_VERIFY") != nullptr;
    static const bool asyncEnv = [&]()
    {
        const char *e = std::getenv("PS2X_GS_ASYNC");
        return !(e && e[0] == '0') && !verifyMode;
    }();
    if (verifyMode)
        m_parDisabled = true;
    m_asyncEnabled = asyncEnv && vram != nullptr;
    if (m_asyncEnabled && !m_async)
        m_async = std::make_unique<AsyncQueue>();
}

void GSCpuBackend::Enqueue(uint32_t type, const void *a, uint32_t aLen, const void *b, uint32_t bLen)
{
    AsyncQueue &q = *m_async;
    const uint32_t len = aLen + bLen;
    const uint64_t need = 8u + ((static_cast<uint64_t>(len) + 7u) & ~7ull);
    if (need > AsyncQueue::kCapacity / 2u)
    {
        // Oversized command: drain and execute on this thread (same order, same result).
        std::lock_guard<std::mutex> plk(q.produceMutex);
        WaitIdle();
        std::vector<uint8_t> tmp(len);
        if (aLen)
            std::memcpy(tmp.data(), a, aLen);
        if (bLen)
            std::memcpy(tmp.data() + aLen, b, bLen);
        std::lock_guard<std::mutex> lock(m_mutex);
        ExecuteCommand(type, tmp.data(), len);
        return;
    }

    std::lock_guard<std::mutex> plk(q.produceMutex);
    if (!q.started)
    {
        q.started = true;
        q.worker = std::thread([this]()
                               { WorkerMain(); });
    }

    uint64_t h = q.head.load(std::memory_order_relaxed);
    uint64_t pos = h % AsyncQueue::kCapacity;
    const uint64_t skip = (pos + need > AsyncQueue::kCapacity) ? AsyncQueue::kCapacity - pos : 0u;

    // Wait for space.
    for (int spins = 0; (h + skip + need) - q.tail.load(std::memory_order_acquire) > AsyncQueue::kCapacity; ++spins)
    {
        if (spins < 64)
        {
            std::this_thread::yield();
            continue;
        }
        std::unique_lock<std::mutex> lk(q.waitMutex);
        q.doneWaiters.fetch_add(1);
        q.doneCv.wait_for(lk, std::chrono::milliseconds(1), [&]()
                          { return (h + skip + need) - q.tail.load(std::memory_order_acquire) <= AsyncQueue::kCapacity; });
        q.doneWaiters.fetch_sub(1);
    }

    uint8_t *buf = q.buffer.get();
    if (skip)
    {
        const uint32_t wrapHdr[2] = {AsyncQueue::kWrap, 0u};
        std::memcpy(buf + pos, wrapHdr, 8);
        h += skip;
        pos = 0;
    }
    const uint32_t hdr[2] = {type, len};
    std::memcpy(buf + pos, hdr, 8);
    if (aLen)
        std::memcpy(buf + pos + 8, a, aLen);
    if (bLen)
        std::memcpy(buf + pos + 8 + aLen, b, bLen);
    q.head.store(h + need, std::memory_order_seq_cst);
    if (q.workerSleeping.load(std::memory_order_seq_cst))
    {
        std::lock_guard<std::mutex> lk(q.waitMutex);
        q.workCv.notify_one();
    }
}

void GSCpuBackend::WaitIdle() const
{
    if (!m_async || !m_async->started)
        return;
    AsyncQueue &q = *m_async;
    const uint64_t target = q.head.load(std::memory_order_seq_cst);
    for (int spins = 0; q.tail.load(std::memory_order_acquire) < target; ++spins)
    {
        if (spins < 256)
        {
            std::this_thread::yield();
            continue;
        }
        std::unique_lock<std::mutex> lk(q.waitMutex);
        q.doneWaiters.fetch_add(1);
        q.doneCv.wait_for(lk, std::chrono::milliseconds(1), [&]()
                          { return q.tail.load(std::memory_order_acquire) >= target; });
        q.doneWaiters.fetch_sub(1);
    }
}

void GSCpuBackend::WorkerMain()
{
    AsyncQueue &q = *m_async;
    uint8_t *buf = q.buffer.get();
    uint64_t t = q.tail.load(std::memory_order_relaxed);
    for (;;)
    {
        uint64_t h = q.head.load(std::memory_order_acquire);
        if (h == t)
        {
            for (int spins = 0; spins < 2000 && h == t; ++spins)
            {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#endif
                h = q.head.load(std::memory_order_acquire);
            }
            if (h == t)
            {
                std::unique_lock<std::mutex> lk(q.waitMutex);
                q.workerSleeping.store(true, std::memory_order_seq_cst);
                while (q.head.load(std::memory_order_seq_cst) == t && !q.stop.load())
                    q.workCv.wait(lk);
                q.workerSleeping.store(false, std::memory_order_seq_cst);
                if (q.stop.load() && q.head.load() == t)
                    return;
                continue;
            }
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            int executed = 0;
            while (t != h && executed < 256)
            {
                const uint64_t pos = t % AsyncQueue::kCapacity;
                uint32_t hdr[2];
                std::memcpy(hdr, buf + pos, 8);
                if (hdr[0] == AsyncQueue::kWrap)
                {
                    t += AsyncQueue::kCapacity - pos;
                    continue;
                }
                ExecuteCommand(hdr[0], buf + pos + 8, hdr[1]);
                t += 8u + ((static_cast<uint64_t>(hdr[1]) + 7u) & ~7ull);
                ++executed;
            }
            // Queue drained: finish any pending parallel raster batch before going idle.
            if (t == h && q.head.load(std::memory_order_acquire) == t)
                FlushParallel();
        }
        q.tail.store(t, std::memory_order_seq_cst);
        if (q.doneWaiters.load(std::memory_order_seq_cst) > 0)
        {
            std::lock_guard<std::mutex> lk(q.waitMutex);
            q.doneCv.notify_all();
        }
    }
}

void GSCpuBackend::ExecuteCommand(uint32_t type, const uint8_t *payload, uint32_t len)
{
    switch (type)
    {
    case GSCapture::kRecSubmit:
    {
        GSPrimitiveBatch batch;
        std::memcpy(static_cast<void *>(&batch), payload, sizeof(batch));
        ExecSubmit(batch);
        break;
    }
    case GSCapture::kRecLoadClut:
    {
        GSTex0Reg tex0;
        GSTexClutReg clut;
        std::memcpy(static_cast<void *>(&tex0), payload, sizeof(tex0));
        std::memcpy(static_cast<void *>(&clut), payload + sizeof(tex0), sizeof(clut));
        ExecLoadClut(tex0, clut);
        break;
    }
    case GSCapture::kRecBeginTransfer:
    {
        GSTransferCommand cmd;
        std::memcpy(static_cast<void *>(&cmd), payload, sizeof(cmd));
        ExecBeginTransfer(cmd);
        break;
    }
    case GSCapture::kRecUpload:
        ExecUploadImage(payload, len);
        break;
    case GSCapture::kRecTextureFlush:
        ExecTextureFlush();
        break;
    case GSCapture::kRecClear:
    {
        GSContext ctx;
        uint32_t rgba;
        std::memcpy(static_cast<void *>(&ctx), payload, sizeof(ctx));
        std::memcpy(&rgba, payload + sizeof(ctx), sizeof(rgba));
        ExecClearFramebuffer(ctx, rgba);
        break;
    }
    case GSCapture::kRecWriteVram:
    {
        uint32_t a[6];
        std::memcpy(a, payload, sizeof(a));
        ExecWriteVram(a[0], a[1], a[2], a[3], a[4], a[5]);
        break;
    }
    case GSCapture::kRecReset:
        ExecReset();
        break;
    default:
        break;
    }
}

void GSCpuBackend::Submit(const GSPrimitiveBatch &batch)
{
    if (AsyncActive())
    {
        Enqueue(GSCapture::kRecSubmit, &batch, sizeof(batch));
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    ExecSubmit(batch);
}

void GSCpuBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    if (AsyncActive())
    {
        Enqueue(GSCapture::kRecLoadClut, &tex0, sizeof(tex0), &texclut, sizeof(texclut));
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    ExecLoadClut(tex0, texclut);
}

void GSCpuBackend::BeginTransfer(const GSTransferCommand &command)
{
    if (AsyncActive())
    {
        Enqueue(GSCapture::kRecBeginTransfer, &command, sizeof(command));
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    ExecBeginTransfer(command);
}

void GSCpuBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes == 0u)
        return;
    if (AsyncActive())
    {
        Enqueue(GSCapture::kRecUpload, data, sizeBytes);
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    ExecUploadImage(data, sizeBytes);
}

void GSCpuBackend::TextureFlush()
{
    if (AsyncActive())
    {
        Enqueue(GSCapture::kRecTextureFlush, nullptr, 0);
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    ExecTextureFlush();
}

void GSCpuBackend::Reset()
{
    if (AsyncActive())
    {
        Enqueue(GSCapture::kRecReset, nullptr, 0);
        WaitIdle();
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    ExecReset();
}

bool GSCpuBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    if (AsyncActive())
    {
        // Same result as ExecClearFramebuffer, which only depends on the arguments here.
        const bool handled = m_vram && context.frame.fbw != 0u &&
                             (context.frame.psm == GS_PSM_CT32 || context.frame.psm == GS_PSM_CT24 ||
                              context.frame.psm == GS_PSM_CT16 || context.frame.psm == GS_PSM_CT16S);
        Enqueue(GSCapture::kRecClear, &context, sizeof(context), &rgba, sizeof(rgba));
        return handled;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    return ExecClearFramebuffer(context, rgba);
}

void GSCpuBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    if (AsyncActive())
    {
        const uint32_t args[6] = {psm, base, bw, x, y, value};
        Enqueue(GSCapture::kRecWriteVram, args, sizeof(args));
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    ExecWriteVram(psm, base, bw, x, y, value);
}

uint32_t GSCpuBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    WaitIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    return ExecConsumeLocalToHostBytes(dst, maxBytes);
}

uint32_t GSCpuBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    WaitIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    const_cast<GSCpuBackend *>(this)->FlushParallel();
    return ReadVramUnlocked(psm, base, bw, x, y);
}

void GSCpuBackend::Flush()
{
}

void GSCpuBackend::Sync(GSSyncReason reason)
{
    // FINISH needs no VRAM visibility: every observer of backend state drains the queue itself.
    if (reason != GSSyncReason::Finish)
        WaitIdle();
}
