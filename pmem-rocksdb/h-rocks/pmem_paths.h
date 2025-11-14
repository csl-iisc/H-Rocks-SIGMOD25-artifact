#pragma once

#include <cstdlib>
#include <string>

namespace hrocks {

inline std::string NormalizeBase(const char* env_name, const char* default_path) {
  const char* from_env = std::getenv(env_name);
  std::string base = (from_env != nullptr && from_env[0] != '\0') ? from_env : default_path;
  while (!base.empty() && base.back() == '/') {
    base.pop_back();
  }
  return base.empty() ? std::string(default_path) : base;
}

inline const std::string& PmemBase() {
  static const std::string base = NormalizeBase("HR_PMEM_DIR", "/pmem");
  return base;
}

inline const std::string& ShmBase() {
  static const std::string base = NormalizeBase("HR_SHM_DIR", "/dev/shm");
  return base;
}

inline std::string JoinPath(const std::string& base, const std::string& child) {
  if (child.empty()) return base;
  if (!child.empty() && child.front() == '/') return child;
  if (base.empty()) return child;
  return base + "/" + child;
}

inline std::string PmemPath(const std::string& child) { return JoinPath(PmemBase(), child); }
inline std::string ShmPath(const std::string& child) { return JoinPath(ShmBase(), child); }

}  // namespace hrocks

