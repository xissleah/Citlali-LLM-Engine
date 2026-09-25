#pragma once

#include "citlali/io/gguf_reader.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace citlali::remote {

enum class Backend { Auto, Gpu, Npu, Hybrid };
enum class Transport { Usb, Wifi6 };

struct RemoteOptions {
    std::string endpoint;
    Transport transport = Transport::Usb;
    Backend backend = Backend::Gpu;
    std::vector<std::uint32_t> layers;

    bool enabled() const {
        return !endpoint.empty() && !layers.empty();
    }
    bool offloads_layer(std::uint32_t layer_index) const;
};

class RemoteClient {
public:
    explicit RemoteClient(const RemoteOptions& options);
    ~RemoteClient();

    RemoteClient(const RemoteClient&) = delete;
    RemoteClient& operator=(const RemoteClient&) = delete;

    void upload_layer(const io::GgufFile& gguf,
                      std::uint32_t layer_index);
    std::vector<std::uint16_t> run_layer_range(
        std::uint32_t first_layer,
        std::uint32_t last_layer,
        std::uint32_t position,
        std::uint32_t max_context,
        const std::vector<std::uint16_t>& input,
        std::uint32_t ffn_elements,
        std::uint32_t heads,
        std::uint32_t kv_heads,
        std::uint32_t head_dim,
        float rms_norm_epsilon,
        float rope_theta);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace citlali::remote
