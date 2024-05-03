#include "mrc/node/rx_source.hpp"

namespace mrc::node {

template class RxSource<int, runnable::Context>;
template class RxSource<float, runnable::Context>;
template class RxSource<double, runnable::Context>;
template class RxSource<std::string, runnable::Context>;

}  // namespace mrc::node
