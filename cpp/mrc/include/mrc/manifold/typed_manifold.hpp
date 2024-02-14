#pragma once

#include "mrc/channel/buffered_channel.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/edge/edge_builder.hpp"
#include "mrc/edge/edge_writable.hpp"
#include "mrc/manifold/manifold.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/operators/node_component.hpp"
#include "mrc/node/operators/router.hpp"
#include "mrc/node/sink_channel_owner.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/runtime/remote_descriptor.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/string_utils.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>

namespace mrc::manifold {

struct ManifoldAction
{
    PortAddress2 source_address;
    std::unique_ptr<runtime::ValueDescriptor> descriptor;
    std::function<void()> on_completed;
};

template <typename T>
class ManifoldTagger2 : public ManifoldTaggerBase2,
                        // Mimic a RxSink Base
                        public node::ReadableWritableSink<ManifoldAction>,
                        // Channel to hold data until we have downstream connections
                        public node::SinkChannelOwner<ManifoldAction>,
                        // Half of a router node to write to multiple outputs
                        public node::RouterWritableAcceptor<PortAddress2, std::unique_ptr<runtime::ValueDescriptor>>
{
  public:
    using key_t            = PortAddress2;
    using input_message_t  = ManifoldAction;
    using output_message_t = std::pair<PortAddress2, std::unique_ptr<runtime::ValueDescriptor>>;

    ManifoldTagger2()
    {
        // Set the default channel
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<input_message_t>>());
    }

    void add_output(PortAddress2 port_address, bool is_local, edge::IWritableProviderBase* output_sink) override
    {
        mrc::make_edge(*this->get_source(port_address), *output_sink);
    }

  protected:
    channel::Status on_next(input_message_t&& data)
    {
        // Move it so it goes out of scope
        auto local_data = std::move(data);

        channel::Status status = channel::Status::success;

        if (local_data.descriptor)
        {
            // Figure out which port to send this to
            auto port = this->get_next_tag();

            // auto output = this->convert_value(std::move(data));

            status = this->get_writable_edge(port)->await_write(std::move(local_data.descriptor));
        }

        // If there is a completion function, call it
        if (local_data.on_completed)
        {
            local_data.on_completed();
        }

        return status;
    }

  private:
    channel::Status process_one_message() override
    {
        input_message_t data;

        // Now pull from the queue. Dont wait for any time if there isnt a message
        auto status = this->get_readable_edge()->await_read_for(data, channel::duration_t::zero());

        if (status == channel::Status::success)
        {
            return this->on_next(std::move(data));
        }

        return status;
    }
};

// template <typename T>
// class ManifoldTagger : public ManifoldTaggerBase,
//                        public node::WritableProvider<T>,
//                        public node::SinkChannelOwner<T>,
//                        public node::RouterWritableAcceptor<SegmentAddress, std::pair<SegmentAddress, T>>
// {
//   public:
//     using input_message_t  = T;
//     using output_message_t = std::pair<SegmentAddress, T>;

//     ManifoldTagger()
//     {
//         // Set the default channel
//         this->set_channel(std::make_unique<mrc::channel::BufferedChannel<input_message_t>>());

//         // To prevent closing the input channel when all segments go away, create an entrypoint ourselves to
//         guarantee
//         // lifetime
//         m_tagger_entrypoint = std::make_shared<node::WritableEntrypoint<input_message_t>>();

//         // Connect to ourselves
//         mrc::make_edge(*m_tagger_entrypoint, *this);
//     }

//   private:
//     void drop_outputs() override
//     {
//         this->drop_all_sources();
//     }

//     edge::IWritableAcceptorBase& get_output(SegmentAddress address) const override
//     {
//         return *this->get_source(address);
//     }

//     channel::Status process_one() override
//     {
//         input_message_t data;

//         auto readable_edge = this->get_readable_edge();

//         auto status = readable_edge->await_read_for(data, channel::duration_t::zero());

//         if (status == channel::Status::success)
//         {
//             auto tag = this->get_next_tag();

//             this->get_writable_edge(tag)->await_write(std::make_pair(tag, std::move(data)));
//         }

//         return status;
//     }

//     // This entrypoint is used to keep the channel open incase all upstream egress points go away. Also, it allows
//     for
//     // errored messages to be re-tagged so they arent lost
//     std::shared_ptr<node::WritableEntrypoint<input_message_t>> m_tagger_entrypoint;
// };

// template <typename T>
// class ManifoldUnTagger : public ManifoldUnTaggerBase,
//                          public node::WritableProvider<std::pair<SegmentAddress, T>>,
//                          public node::SinkChannelOwner<std::pair<SegmentAddress, T>>,
//                          public node::RouterWritableAcceptor<SegmentAddress, T>
// {
//   public:
//     using input_message_t  = std::pair<SegmentAddress, T>;
//     using output_message_t = T;

//     ManifoldUnTagger()
//     {
//         // Set the default channel
//         this->set_channel(std::make_unique<mrc::channel::BufferedChannel<input_message_t>>());

//         // To prevent closing the input channel when all segments go away, create an entrypoint ourselves to
//         guarantee
//         // lifetime
//         m_tagger_entrypoint = std::make_shared<node::WritableEntrypoint<input_message_t>>();

//         // Connect to ourselves
//         mrc::make_edge(*m_tagger_entrypoint, *this);
//     }

//   private:
//     void drop_outputs() override
//     {
//         return this->drop_all_sources();
//     }

//     edge::IWritableAcceptorBase& get_output(SegmentAddress address) const override
//     {
//         return *this->get_source(address);
//     }

//     channel::Status process_one() override
//     {
//         input_message_t data;

//         auto status = this->get_readable_edge()->await_read_for(data, channel::duration_t::zero());

//         if (status == channel::Status::success)
//         {
//             // Use the tag to determine where it should go
//             auto tag = data.first;

//             this->get_writable_edge(tag)->await_write(std::move(data.second));
//         }

//         return status;
//     }

//     // This entrypoint is used to keep the channel open incase all upstream egress points go away. Also, it allows
//     for
//     // errored messages to be re-tagged so they arent lost
//     std::shared_ptr<node::WritableEntrypoint<input_message_t>> m_tagger_entrypoint;
// };

template <typename T>
class TypedManifold : public ManifoldBase
{
  public:
    TypedManifold(runnable::IRunnableResources& resources, std::string port_name) :
      ManifoldBase(resources, std::move(port_name), std::make_unique<ManifoldTagger2<T>>())
    {}
};
}  // namespace mrc::manifold
