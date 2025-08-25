#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/transform.h>

struct custom_key_type {
    uint64_t Key;

    __host__ __device__ custom_key_type() {}
    __host__ __device__ custom_key_type(uint64_t x) : Key{x} {}

    // Device equality operator is mandatory due to libcudacxx bug:
    // https://github.com/NVIDIA/libcudacxx/issues/223
    __device__ bool operator==(custom_key_type const& other) const
    {
        //return a == other.a and b == other.b and c == other.c;
        return Key == other.Key; 
    }
};

struct custom_value_type {
    uint64_t key_idx;
    uint64_t value_idx;
    uint64_t int_counter;

    __host__ __device__ custom_value_type() {}
    __host__ __device__ custom_value_type(uint64_t x) : key_idx{x}, value_idx{x}, int_counter{x} {}
    __device__ bool operator==(custom_value_type const& other) const
    {
        return key_idx == other.key_idx and value_idx == other.value_idx and int_counter == other.int_counter;
    }
};

namespace cuco {
    template <>
        struct is_bitwise_comparable<custom_value_type> : std::true_type {};

} 
// User-defined device hash callable
struct custom_hash {
    template <typename key_type>
        __device__ uint64_t operator()(key_type k)
        {
            return k.Key;
        };
};

// User-defined device key equal callable
struct custom_key_equals {
    template <typename key_type>
        __device__ bool operator()(key_type const& lhs, key_type const& rhs)
        {
            return lhs.Key == rhs.Key;
        }
};


