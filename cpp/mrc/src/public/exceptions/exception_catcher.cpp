#include <mrc/exceptions/exception_catcher.hpp>

namespace mrc {

void ExceptionCatcher::push_exception(std::exception_ptr ex)
{
    auto lock = std::lock_guard(m_mutex);
    m_exceptions.push(ex);
}

bool ExceptionCatcher::has_exception()
{
    auto lock = std::lock_guard(m_mutex);
    return not m_exceptions.empty();
}

void ExceptionCatcher::rethrow_next_exception()
{
    auto lock = std::lock_guard(m_mutex);

    if (m_exceptions.empty())
    {
        return;
    }

    auto ex = m_exceptions.front();

    m_exceptions.pop();

    std::rethrow_exception(ex);
}

}  // namespace mrc
