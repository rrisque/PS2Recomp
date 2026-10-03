#pragma once

// GS backend call capture (debug/verification tool).
//
// PS2X_GS_CAPTURE=<file>            enable; the stream is written to <file>
// PS2X_GS_CAPTURE_START=<seconds>   start this many seconds after the first backend call (default 0)
// PS2X_GS_CAPTURE_SECS=<seconds>    capture duration (default 30)
// PS2X_GS_CAPTURE_MAX_MB=<mb>       stop once the file reaches this size (default 1024)
//
// The file starts with the complete backend state (VRAM, CLUT, texture page cache, transfer
// state) followed by every GSRasterBackend call in the order the backend executed it. The
// replay tool feeds the stream into a backend and checks VRAM / presented frames bit-exactly.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace GSCapture
{
    enum RecordType : uint32_t
    {
        kRecState = 1,
        kRecSubmit = 2,
        kRecLoadClut = 3,
        kRecBeginTransfer = 4,
        kRecUpload = 5,
        kRecTextureFlush = 6,
        kRecPresent = 7,
        kRecClear = 8,
        kRecConsume = 9,
        kRecWriteVram = 10,
        kRecReset = 11,
        kRecEnd = 12,
    };

    static constexpr char kMagic[8] = {'G', 'S', 'C', 'A', 'P', '0', '0', '1'};

    inline uint64_t Hash64(const void *data, size_t size, uint64_t h = 0xcbf29ce484222325ull)
    {
        // FNV-1a over 8-byte words (tail bytewise); only used for equality checks.
        const uint8_t *p = static_cast<const uint8_t *>(data);
        size_t i = 0;
        for (; i + 8 <= size; i += 8)
        {
            uint64_t w;
            std::memcpy(&w, p + i, 8);
            h ^= w;
            h *= 0x100000001b3ull;
            h ^= h >> 29;
        }
        for (; i < size; ++i)
        {
            h ^= p[i];
            h *= 0x100000001b3ull;
        }
        return h;
    }

    class Writer
    {
    public:
        static Writer &Instance()
        {
            static Writer w;
            return w;
        }

        bool Enabled() const { return m_enabled; }

        // Called under the backend lock before each recorded call. Returns true when the call
        // must be recorded; sets needState when the initial state record must be written first.
        bool Tick(bool &needState)
        {
            needState = false;
            if (!m_enabled || m_done)
                return false;
            const auto now = std::chrono::steady_clock::now();
            if (!m_clockStarted)
            {
                m_clockStarted = true;
                m_t0 = now;
            }
            const double t = std::chrono::duration<double>(now - m_t0).count();
            if (!m_file)
            {
                if (t < m_startSecs)
                    return false;
                m_file = std::fopen(m_path, "wb");
                if (!m_file)
                {
                    m_done = true;
                    return false;
                }
                std::fwrite(kMagic, 1, sizeof(kMagic), m_file);
                m_tStart = t;
                needState = true;
                std::fprintf(stderr, "[gs-capture] started -> %s\n", m_path);
                return true;
            }
            if (t - m_tStart >= m_durationSecs || m_bytes >= m_maxBytes)
            {
                Record(kRecEnd, nullptr, 0);
                std::fclose(m_file);
                m_file = nullptr;
                m_done = true;
                std::fprintf(stderr, "[gs-capture] finished (%llu bytes, %llu records)\n",
                             (unsigned long long)m_bytes, (unsigned long long)m_records);
                return false;
            }
            return true;
        }

        void Record(uint32_t type, const void *a, size_t aLen, const void *b = nullptr, size_t bLen = 0)
        {
            if (!m_file)
                return;
            const uint32_t hdr[2] = {type, static_cast<uint32_t>(aLen + bLen)};
            std::fwrite(hdr, sizeof(hdr), 1, m_file);
            if (aLen)
                std::fwrite(a, 1, aLen, m_file);
            if (bLen)
                std::fwrite(b, 1, bLen, m_file);
            m_bytes += sizeof(hdr) + aLen + bLen;
            ++m_records;
        }

    private:
        Writer()
        {
            const char *path = std::getenv("PS2X_GS_CAPTURE");
            if (!path || !*path)
                return;
            std::snprintf(m_path, sizeof(m_path), "%s", path);
            if (const char *s = std::getenv("PS2X_GS_CAPTURE_START"))
                m_startSecs = std::atof(s);
            if (const char *s = std::getenv("PS2X_GS_CAPTURE_SECS"))
                m_durationSecs = std::atof(s);
            if (const char *s = std::getenv("PS2X_GS_CAPTURE_MAX_MB"))
                m_maxBytes = static_cast<uint64_t>(std::atof(s) * 1048576.0);
            m_enabled = true;
        }

        char m_path[1024] = {};
        bool m_enabled = false;
        bool m_done = false;
        bool m_clockStarted = false;
        std::chrono::steady_clock::time_point m_t0{};
        double m_tStart = 0.0;
        double m_startSecs = 0.0;
        double m_durationSecs = 30.0;
        uint64_t m_maxBytes = 1024ull * 1048576ull;
        uint64_t m_bytes = 0;
        uint64_t m_records = 0;
        FILE *m_file = nullptr;
    };
}
