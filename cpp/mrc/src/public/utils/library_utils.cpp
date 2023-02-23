#include "mrc/utils/library_utils.hpp"

#include <dlfcn.h>

#include <filesystem>

namespace mrc::utils {

std::filesystem::path get_exe_location()
{
    return std::filesystem::canonical("/proc/self/exe");
}

std::string get_mrc_lib_location()
{
    Dl_info dl_info;
    dladdr((void*)get_mrc_lib_location, &dl_info);

    return dl_info.dli_fname;
}

}  // namespace mrc::utils
