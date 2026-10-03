// HLE for the Team Ico/Japan Studio "usbSerialSys" debug console (PL2303.IRX RPC server).
// The SotC loader spins in usbSerialSysInit() until this server binds; the game then sends its
// debug printf output through it, which we forward to the host log.
#include "module_factories.h"
#include "rpc_reply.h"

#include <array>
#include <mutex>
#include <string>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kUsbSerialSid = 0x80000220u;
        constexpr uint32_t kFuncSend = 0x10u;

        class UsbSerialService final : public IopService
        {
        public:
            explicit UsbSerialService(IopHost &host) : m_host(host) {}
            [[nodiscard]] std::string_view name() const override { return "usbserial"; }
            [[nodiscard]] std::span<const uint32_t> sids() const override { return kSids; }
            [[nodiscard]] std::span<const std::string_view> moduleAliases() const override { return kAliases; }
            void reset() override { std::lock_guard<std::mutex> lock(m_mutex); m_line.clear(); }

            [[nodiscard]] RpcResult handleRpc(const RpcRequest &request) override
            {
                RpcResult result;
                if (request.sid != kUsbSerialSid)
                    return result;
                result.handled = true;
                result.resultAddress = request.receive.address;
                if (request.function == kFuncSend && request.send.address && request.send.size)
                {
                    std::array<char, 0x41> buf{};
                    const size_t n = std::min<size_t>(request.send.size, 0x40);
                    if (m_host.readGuest(request.send.address, buf.data(), n))
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        for (size_t i = 0; i < n && buf[i]; ++i)
                        {
                            const char c = buf[i];
                            if (c == '\r' || c == '\n')
                            {
                                if (!m_line.empty())
                                    m_host.log(LogLevel::Info, "[SotC] " + m_line);
                                m_line.clear();
                            }
                            else
                                m_line.push_back(c);
                        }
                    }
                    const std::array<uint32_t, 1> ok{0u};
                    (void)writeRpcWords(m_host, request.receive, ok);
                }
                return result;
            }

        private:
            inline static constexpr std::array<uint32_t, 1> kSids{kUsbSerialSid};
            inline static constexpr std::array<std::string_view, 2> kAliases{"pl2303", "usbserial"};
            IopHost &m_host;
            std::mutex m_mutex;
            std::string m_line;
        };
    }

    std::unique_ptr<IopService> createUsbSerialService(IopHost &host)
    {
        return std::make_unique<UsbSerialService>(host);
    }
}
