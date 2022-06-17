#include <srf/utils/string_utils.hpp>

#include <string>

namespace srf::quickstart::hybrid::common {

struct DataObject
{
    DataObject(std::string n = "", int v = 0) : name(std::move(n)), value(v) {}

    std::string to_string() const
    {
        return SRF_CONCAT_STR("{Name: '" << this->name << "', Value: " << this->value << "}");
    }

    std::string name;
    int value{0};
};
}  // namespace srf::quickstart::hybrid::common
