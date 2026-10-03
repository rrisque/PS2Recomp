#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>

namespace GSMem
{
    // Single-page texture cache: texture reads see the contents a page had when it was last
    // fetched (miss), until the next miss or TEXFLUSH.
    //
    // The copy is taken lazily: a miss only records the page, and reads go straight to VRAM
    // while that page is unchanged. The copy is materialised right before the first VRAM
    // write that could touch the page (NoteWrite / PrepareUnknownWrite), so reads observe
    // exactly the bytes an eager copy at miss time would have produced.
    class TexturePageCache
    {
    public:
        static constexpr uint32_t kPageSize = 8192u;
        static constexpr uint32_t kNoPage = UINT32_MAX;

        void Invalidate() noexcept
        {
            m_pageBase = kNoPage;
            m_materialized = false;
            m_watchPage = kNoPage;
        }

        static bool Disabled() noexcept
        {
            static const bool disabled = std::getenv("PS2X_GS_NO_TEXCACHE") != nullptr;
            return disabled;
        }

        // byteAddress is the wrapped, swizzled VRAM address. The returned
        // pointer is valid only until the next miss, write or invalidation.
        const uint8_t *Resolve(const uint8_t *vram, uint32_t byteAddress) noexcept
        {
            const uint32_t pageBase = byteAddress & ~(kPageSize - 1u);
            if (m_pageBase != pageBase)
            {
                if (Disabled())
                    return vram + byteAddress;
                m_pageBase = pageBase;
                m_materialized = false;
                m_watchPage = pageBase;
                return vram + byteAddress;
            }
            return m_materialized ? m_bytes.data() + (byteAddress & (kPageSize - 1u)) : vram + byteAddress;
        }

        // Must be called before writing VRAM at byteAddress (any address inside the written word).
        inline void NoteWrite(const uint8_t *vram, uint32_t byteAddress) noexcept
        {
            if ((byteAddress & ~(kPageSize - 1u)) == m_watchPage)
                Materialize(vram);
        }

        // Must be called before a VRAM write whose address is not known.
        inline void PrepareUnknownWrite(const uint8_t *vram) noexcept
        {
            if (m_watchPage != kNoPage)
                Materialize(vram);
        }

        // A copy identical to the VRAM page behaves exactly like the pending (not yet copied)
        // state of the same page: reads see the same bytes, and the copy a later write to the page
        // would take holds those same bytes. Returns to the pending state in that case.
        void Unmaterialize(const uint8_t *vram) noexcept
        {
            if (m_materialized && m_pageBase != kNoPage && vram &&
                std::memcmp(m_bytes.data(), vram + m_pageBase, kPageSize) == 0)
            {
                m_materialized = false;
                m_watchPage = m_pageBase;
            }
        }

        // Page whose writes must be reported (kNoPage when no copy is pending).
        uint32_t WatchPage() const noexcept { return m_watchPage; }
        uint32_t PageBase() const noexcept { return m_pageBase; }
        bool Materialized() const noexcept { return m_materialized; }

        // Cache contents for serialisation (the VRAM page while the copy is still pending).
        const uint8_t *Bytes(const uint8_t *vram) const noexcept
        {
            if (m_materialized || m_pageBase == kNoPage || !vram)
                return m_bytes.data();
            return vram + m_pageBase;
        }

        // State after reading pageBase with no pending copy (as left by a miss on that page).
        void SetPending(uint32_t pageBase) noexcept
        {
            m_pageBase = pageBase;
            m_materialized = false;
            m_watchPage = pageBase;
        }

        void Restore(uint32_t pageBase, const uint8_t *bytes) noexcept
        {
            m_pageBase = pageBase;
            std::memcpy(m_bytes.data(), bytes, kPageSize);
            m_materialized = true;
            m_watchPage = kNoPage;
        }

    private:
        void Materialize(const uint8_t *vram) noexcept
        {
            std::memcpy(m_bytes.data(), vram + m_pageBase, kPageSize);
            m_materialized = true;
            m_watchPage = kNoPage;
        }

        alignas(64) std::array<uint8_t, kPageSize> m_bytes{};
        uint32_t m_pageBase = kNoPage;
        uint32_t m_watchPage = kNoPage;
        bool m_materialized = false;
    };
}
