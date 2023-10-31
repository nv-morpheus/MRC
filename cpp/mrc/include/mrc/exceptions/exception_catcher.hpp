#include <exception>
#include <mutex>
#include <queue>

namespace mrc {

/**
 * @brief A utility for catching out-of-stack exceptions in a thread-safe manner such that they
 * can be checked and throw from a parent thread.
 */
class ExceptionCatcher
{
  public:
    /**
     * @brief "catches" an exception to the catcher
     */
    void push_exception(std::exception_ptr ex);

    /**
     * @brief checks to see if any exceptions have been "caught" by the catcher.
     */
    bool has_exception();

    /**
     * @brief rethrows the next exception (in the order in which it was "caught").
     */
    void rethrow_next_exception();

  private:
    std::mutex m_mutex{};
    std::queue<std::exception_ptr> m_exceptions{};
};

}  // namespace mrc
