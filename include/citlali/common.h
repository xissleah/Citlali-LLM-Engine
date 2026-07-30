#pragma once

#include <stdexcept>
#include <string>

namespace citlali {

    class Error : public std::runtime_error {
    public:
        explicit Error(const std::string& message) : std::runtime_error(message) {}
    };

    inline void require(bool condition, const std::string& message) {
        if (!condition) {
            throw Error(message);
        }
    }

} // namespace citlali
