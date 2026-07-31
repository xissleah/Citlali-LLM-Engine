#include "citlali/io/gguf_reader.h"

#include "citlali/common.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

namespace citlali::io {
    namespace {
        template <typename T>
        T read_pod(std::ifstream& input) {
            T value{};
            input.read(reinterpret_cast<char*>(&value), sizeof(T));
            require((bool)input,"gguf文件读取失败");
            return value;
        }
        // 读取单个sizeof(T)，在读取单个数据时可以减少io开销

        std::string read_string(std::ifstream& input)
        {
            const uint64_t size = read_pod<uint64_t>(input);
            require(size<=1uLL<<32,"GGUF String Is Too Large");
            std::string text(size,'\0');
            if (size>0)
            {
                input.read(text.data(),static_cast<std::streamsize>(size));
                require((bool)input,"Failed To Read GGUF String");
            }
            return text;
        }
        // 读取字符串

        template <typename T>
        std::vector<T> read_array(std::ifstream& input, uint64_t count)
        {
            require(count <= (1ull << 32), "GGUF Array Size Too Large");
            std::vector<T> values(static_cast<size_t>(count));
            if (!values.empty())
            {
                input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
                require((bool)input,"Failed To Read GGUF Array Size");
            }
            return values;
        }
        // 批量读取多个sizeof(T)，多个数据一次io读完减少io开销，实际上sizeof(T)==1的时候功能等价于上面的read_pod，但实际上会多创建vector的开销，所以两个分开写

        GgufScalar read_scalar(std::ifstream& input, GgufMetadataType type) {
            switch (type) {
            case GgufMetadataType::Uint8: return read_pod<uint8_t>(input);
            case GgufMetadataType::Int8: return read_pod<int8_t>(input);
            case GgufMetadataType::Uint16: return read_pod<uint16_t>(input);
            case GgufMetadataType::Int16: return read_pod<int16_t>(input);
            case GgufMetadataType::Uint32: return read_pod<uint32_t>(input);
            case GgufMetadataType::Int32: return read_pod<int32_t>(input);
            case GgufMetadataType::Float32: return read_pod<float>(input);
            case GgufMetadataType::Bool: return read_pod<uint8_t>(input) != 0;
            case GgufMetadataType::String: return read_string(input);
            case GgufMetadataType::Uint64: return read_pod<uint64_t>(input);
            case GgufMetadataType::Int64: return read_pod<int64_t>(input);
            case GgufMetadataType::Float64: return read_pod<double>(input);
            default: throw Error("unsupported GGUF metadata scalar type");
            }
        }
        // 根据不同类型单个读取


        GgufArray read_array_value(std::ifstream& input, GgufMetadataType item_type) {
            const uint64_t count = read_pod<uint64_t>(input);
            switch (item_type) {
            case GgufMetadataType::Uint8: return read_array<uint8_t>(input, count);
            case GgufMetadataType::Int8: return read_array<int8_t>(input, count);
            case GgufMetadataType::Uint16: return read_array<uint16_t>(input, count);
            case GgufMetadataType::Int16: return read_array<int16_t>(input, count);
            case GgufMetadataType::Uint32: return read_array<uint32_t>(input, count);
            case GgufMetadataType::Int32: return read_array<int32_t>(input, count);
            case GgufMetadataType::Float32: return read_array<float>(input, count);
            case GgufMetadataType::Bool: {
                    std::vector<uint8_t> raw = read_array<uint8_t>(input, count);
                    std::vector<bool> values;
                    values.reserve(raw.size());
                    for (uint8_t v : raw) values.push_back(v != 0);
                    return values;
            }
            case GgufMetadataType::String: {
                    require(count <= (1ull << 32), "GGUF string array is too large");
                    std::vector<std::string> values;
                    values.reserve(static_cast<size_t>(count));
                    for (uint64_t i = 0; i < count; ++i) {
                        values.push_back(read_string(input));
                    }
                    return values;
            }
            case GgufMetadataType::Uint64: return read_array<uint64_t>(input, count);
            case GgufMetadataType::Int64: return read_array<int64_t>(input, count);
            case GgufMetadataType::Float64: return read_array<double>(input, count);
            default: throw Error("unsupported GGUF metadata array type");
            }
        }
        // 批量读取，由于8个bool会合并为一个字节所以我们要一个个去解码，不同的string长度可能不一致，所以得逐个去push

        GgufValue read_value(std::ifstream& input)
        {
            const auto type = static_cast<GgufMetadataType>(read_pod<uint32_t>(input));
            if (type == GgufMetadataType::Array)    // 只有这个类型需要批量读取
            {
                const auto item_type = static_cast<GgufMetadataType>(read_pod<uint32_t>(input));
                return read_array_value(input, item_type);
            }
            return read_scalar(input, type);
        }
        // 类型判断，这相当于顶层的入口，先用它判断出类型然后用read_scalar/read_array_value进行读取，read_scalar/read_array_value会调用raad_pod/read_string/read_array进行读取数据

        uint64_t align_to(uint64_t value, uint64_t alignment)
        {
            if (alignment == 0) return value;
            return (value + alignment - 1) & ~(alignment - 1);  // -1是为了向上取整
            // 作用同通用性更强的((value+alignment-1)/alignment)*alignment，但仅限于alignment为2的n次方时，考虑到硬件对齐一般都是2的n次方，所以这里位运算合适且更快
        }
        // 这一步是性能优化操作，不是必要操作，硬件读取权重的时候，要求起始地址是某个数的n次方，这样对齐可以提高效率，尤其是在gpu上跑

        template <typename T>
        bool get_scalar_as(const GgufValue& value, T& out) {
            if (const auto* scalar = std::get_if<GgufScalar>(&value)) {
                // 如果不是向量（array）就进入这个if
                if (const auto* exact = std::get_if<T>(scalar)) {
                    // 如果是我想要的元素才进入这个if
                    out = *exact;
                    // 这里的*exact是解引用
                    return true;
                }
            }
            return false;
            // 否则返回false
        }
        // 这是一个安全取值函数，out存的就是取值的东西

        template <typename T>
        std::vector<T> get_array_as(const GgufValue& value) {
            if (const auto* array = std::get_if<GgufArray>(&value)) {
                if (const auto* exact = std::get_if<std::vector<T>>(array)) {
                    return *exact;
                }
            }
            return {};
            // 不满足返回空vector
        }
        // 这个功能和上面类似，只不过这个是取array的
    }

    uint64_t GgufTensorInfo::element_count() const {
        uint64_t count = 1;
        for (uint64_t dim : dims) {
            count *= dim;
        }
        return count;
    }

    void GgufFile::load(const std::string& path)
    {
        path_ = path;
        tensors_.clear();
        metadata_.clear();
        tensor_index_.clear();

        std::ifstream input(path, std::ios::binary);
        // gguf不是文本文件，是二进制格式，所以要用binary打开
        require((bool)input,"Failed to open file: " + path);

        char magic[4]{};
        input.read(magic, 4);
        // 读取文件魔数（文件签名）
        require(std::memcmp(magic, "GGUF", 4) == 0,"no a GGUF file: " + path);
        // GGUF文件的魔数肯定得是GGUF啦

        version_ = read_pod<uint32_t>(input);
        // 版本号 4字节
        const uint64_t tensor_count = read_pod<uint64_t>(input);
        // tensor数量
        const uint64_t metadata_count = read_pod<uint64_t>(input);
        // metadata 数量

        for (uint64_t i = 0 ; i < metadata_count; ++i)
        {
            std::string key = read_string(input);
            metadata_[key] = read_value(input);
        }
        // 循环读取所以metadata
        /*
         * metadata的格式是
         * key
         * value 类型
         * value 内容
         */

        alignment_ = get_u64("general.alignment", 32);
        // 读取tensor数据对齐，如果存在并且是可读取的整数就用metadata的值，否则就默认用32
        // 一类对齐是：gguf文件里tensor数据地址要对齐，比如你一个tensor读到1050读完，它不满足32的倍数，则向上取到1056才是下一个tensor的起始位置,1050到1056之间的值仅用来占位
        // 另一类是：[文件头][metadata][tensor 信息][填充区域][tensor 数据]

        tensors_.reserve(static_cast<size_t>(tensor_count));
        // 给tensor信息留空间

        for (uint64_t i = 0; i < tensor_count; ++i)
        {
            GgufTensorInfo tensor;
            tensor.name = read_string(input);
            // tensor 的名字
            const uint32_t n_dims = read_pod<uint32_t>(input);
            // 读取tensor维度的数量
            tensor.dims.reserve(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d)
            {
                tensor.dims.push_back(read_pod<uint64_t>(input));
            }
            // 读取所有维度
            tensor.type = static_cast<compute::GgufTensorType>(read_pod<uint32_t>(input));
            // 读取tensor类型
            tensor.relative_offset = read_pod<uint64_t>(input);
            // 读取相对偏移量，它不是相对应文件开头而是相对于tensor数据区开头
            tensor.nbytes = compute::gguf_tensor_nbytes(tensor.type, static_cast<size_t>(tensor.element_count()));
            // 计算tensor占用字节数
            require(tensor.nbytes > 0,"unsupported GGUF tensor type for tensor: " + tensor.name + " type=" + compute::to_string(tensor.type));
            // 检查是否支持该tensor
            tensor_index_[tensor.name] = tensors_.size();
            // 建立tensor名称索引，当前tensors_的长度也就是当前tensor在tensors_的下标
            tensors_.push_back(tensor);
        }
        //读取每一个tensor的信息

        const uint64_t current = static_cast<uint64_t>(input.tellg());
        // 获取tensor信息结束位置
        data_offset_ = align_to(current,alignment_);
        // 计算tensor数据区的起点
        for (auto& tensor : tensors_)
        {
            tensor.absolute_offset = data_offset_ + tensor.relative_offset;
        }
        // 计算每一个tensor的绝对偏移
    }

    const GgufTensorInfo* GgufFile::find_tensor(const std::string& name) const
    {
        auto it = tensor_index_.find(name);
        if (it == tensor_index_.end())
        {
            return nullptr;
        }
        return &tensors_[it->second];
    }
    // 根据 tensor 的名称，在已经加载好的 GGUF 文件中查找对应的 tensor 信息，并返回指针；如果找不到，就返回 nullptr

    bool GgufFile::has_key(const std::string& key) const
    {
        return metadata_.find(key) != metadata_.end();
    }
    // 用于判断 metadata_ 中是否存在指定的 key

    std::string GgufFile::get_string(const std::string& key, const std::string& fallback) const
    {
        auto it = metadata_.find(key);
        if (it == metadata_.end())
        {
            return fallback;
        }
        if (const auto* scalar = std::get_if<GgufScalar>(&it->second))
        {
            if (const auto* text = std::get_if<std::string>(scalar))
            {
                return *text;
            }
        }
        return fallback;
    }
    // 从 metadata_ 中读取指定 key 对应的字符串值；如果 key 不存在，或者值不是字符串，就返回 fallback

    uint64_t GgufFile::get_u64(const std::string& key, uint64_t fallback) const
    {
        auto it = metadata_.find(key);
        if (it == metadata_.end())
        {
            return fallback;
        }
        uint64_t u64;
        uint32_t u32;
        int64_t i64;
        int32_t i32;
        if (get_scalar_as(it->second,u64))
        {
            return u64;
        }
        if (get_scalar_as(it->second,u32))
        {
            return u32;
        }
        if (get_scalar_as(it->second,i64))
        {
            return i64 >= 0 ? static_cast<uint64_t>(i64) : fallback;
            // int型有负数，但是unsigned型是无符号的
        }
        if (get_scalar_as(it->second,i32))
        {
            return i32 >= 0 ? static_cast<uint64_t>(i32) : fallback;
        }
        return fallback;
    }
    // 获取数据并转为u64

    int64_t GgufFile::get_i64(const std::string& key, int64_t fallback) const {
        auto it = metadata_.find(key);
        if (it == metadata_.end())
        {
            return fallback;
        }
        int64_t i64;
        int32_t i32;
        uint64_t u64;
        uint32_t u32;
        if (get_scalar_as(it->second, i64))
        {
            return i64;
        }
        if (get_scalar_as(it->second, i32))
        {
            return i32;
        }
        if (get_scalar_as(it->second, u64))
        {
            return static_cast<int64_t>(u64);
        }
        if (get_scalar_as(it->second, u32))
        {
            return static_cast<int64_t>(u32);
        }
        return fallback;
    }
    // 获取数据转为i64

    double GgufFile::get_f64(const std::string& key, double fallback) const {
        auto it = metadata_.find(key);
        if (it == metadata_.end())
        {
            return fallback;
        }
        double f64;
        float f32;
        if (get_scalar_as(it->second, f64))
        {
            return f64;
        }
        if (get_scalar_as(it->second, f32))
        {
            return f32;
        }
        return fallback;
    }
    // 获取数据转double

    float GgufFile::get_f32(const std::string& key, float fallback) const {
        return static_cast<float>(get_f64(key, fallback));
    }
    // 获取数据转float

    bool GgufFile::get_bool(const std::string& key, bool fallback) const {
        auto it = metadata_.find(key);
        if (it == metadata_.end())
        {
            return fallback;
        }
        bool value;
        if (get_scalar_as(it->second, value))
        {
            return value;
        }
        return fallback;
    }
    // 获取数据转bool

    std::vector<std::string> GgufFile::get_string_array(const std::string& key) const {
        auto it = metadata_.find(key);
        if (it == metadata_.end())
        {
            return {};
        }
        return get_array_as<std::string>(it->second);
    }
    // 获取数据转string类型的vector

    std::vector<float> GgufFile::get_f32_array(const std::string& key) const {
        auto it = metadata_.find(key);
        if (it == metadata_.end())
        {
            return {};
        }
        return get_array_as<float>(it->second);
    }
    // 获取数据转float类型的vector

    std::vector<uint32_t> GgufFile::get_u32_array(const std::string& key) const {
        auto it = metadata_.find(key);
        if (it == metadata_.end())
        {
            return {};
        }
        return get_array_as<uint32_t>(it->second);
    }
    // 获取数据转u32类型的vector

    std::vector<uint8_t> GgufFile::read_tensor_bytes(const GgufTensorInfo& tensor) const
    {
        std::ifstream input(path_, std::ios::binary);
        require((bool)input,"failed to open GGUF file: " + path_);
        input.seekg(
            static_cast<std::streamoff>(tensor.absolute_offset),
            std::ios::beg
        );
        require((bool)input,"failed to seek GGUF tensor: " + tensor.name);
        std::vector<uint8_t> bytes(static_cast<size_t>(tensor.nbytes));
        input.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
        require((bool)input,"failed to read GGUF tensor: " + tensor.name);
        return bytes;
    }
    /*
    打开 GGUF 文件
    ↓
    跳转到 tensor.absolute_offset
    ↓
    分配 tensor.nbytes 大小的缓冲区
    ↓
    读取 tensor.nbytes 个字节
    ↓
    返回字节数组
    */

    std::string metadata_value_to_string(const GgufValue& value)
    {
        std::ostringstream out;
        // 创建字符串输出流
        if (const auto* scalar = std::get_if<GgufScalar>(&value)) {
            std::visit([&](const auto& v) { out << v; }, *scalar);
            // [&] 表示按引用捕获当前作用域中使用到的外部变量
            // 类似于printf v这里帮*scalar占位
            return out.str();
        }
        // 为scalar走这条
        if (const auto* array = std::get_if<GgufArray>(&value)) {
            std::visit([&](const auto& v) { out << "array[" << v.size() << "]"; }, *array);
        }
        // 为array走这条
        return out.str();
    }
    //把一个 GgufValue 转换成适合显示或调试的字符串
}
