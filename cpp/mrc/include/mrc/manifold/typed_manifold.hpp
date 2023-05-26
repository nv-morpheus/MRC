#pragma once

#include "mrc/channel/buffered_channel.hpp"
#include "mrc/edge/edge_builder.hpp"
#include "mrc/manifold/manifold.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/operators/node_component.hpp"
#include "mrc/node/operators/router.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/node/writable_entrypoint.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/runnable.hpp"
#include "mrc/types.hpp"

#include <memory>
#include <sstream>

namespace mrc::manifold {

template <typename T>
class ManifoldTagger : public ManifoldTaggerBase,
                       public node::WritableProvider<T>,
                       public node::SinkChannelOwner<T>,
                       public node::RouterWritableAcceptor<SegmentAddress, std::pair<SegmentAddress, T>>
{
  public:
    using input_message_t  = T;
    using output_message_t = std::pair<SegmentAddress, T>;

    ManifoldTagger()
    {
        // Set the default channel
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<input_message_t>>());

        // To prevent closing the input channel when all segments go away, create an entrypoint ourselves to guarantee
        // lifetime
        m_tagger_entrypoint = std::make_shared<node::WritableEntrypoint<input_message_t>>();

        // Connect to ourselves
        mrc::make_edge(*m_tagger_entrypoint, *this);

        m_node_component = std::make_shared<node::LambdaNodeComponent<input_message_t, output_message_t>>(
            [this](input_message_t&& data) -> output_message_t {
                auto tag = this->get_next_tag();

                return std::make_pair(tag, std::move(data));
            });
    }

    // void add_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source) override
    // {
    //     mrc::make_edge(*input_source, *m_node_component);
    // }

    // void add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink) override
    // {
    //     mrc::make_edge(*m_node_component, *output_sink);
    // }

  private:
    edge::IWritableAcceptorBase& get_output(SegmentAddress address) const override
    {
        return *this->get_source(address);
    }

    channel::Status process_one() override
    {
        input_message_t data;

        auto readable_edge = this->get_readable_edge();

        auto status = readable_edge->await_read_for(data, channel::duration_t::zero());

        if (status == channel::Status::success)
        {
            auto tag = this->get_next_tag();

            this->get_writable_edge(tag)->await_write(std::make_pair(tag, std::move(data)));
        }

        return status;
    }

    // This entrypoint is used to keep the channel open incase all upstream egress points go away. Also, it allows for
    // errored messages to be re-tagged so they arent lost
    std::shared_ptr<node::WritableEntrypoint<input_message_t>> m_tagger_entrypoint;

    std::shared_ptr<node::LambdaNodeComponent<input_message_t, output_message_t>> m_node_component;
};

template <typename T>
class ManifoldUnTagger : public ManifoldUnTaggerBase,
                         public node::WritableProvider<std::pair<SegmentAddress, T>>,
                         public node::SinkChannelOwner<std::pair<SegmentAddress, T>>,
                         public node::RouterWritableAcceptor<SegmentAddress, T>
{
  public:
    using input_message_t  = std::pair<SegmentAddress, T>;
    using output_message_t = T;

    ManifoldUnTagger()
    {
        // Set the default channel
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<input_message_t>>());

        m_node_component = std::make_shared<node::LambdaNodeComponent<input_message_t, output_message_t>>(
            [this](input_message_t&& data) -> output_message_t {
                return std::move(data.second);
            });
    }

    // void add_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source) override
    // {
    //     mrc::make_edge(*input_source, *m_node_component);
    // }

    // void add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink) override
    // {
    //     mrc::make_edge(*m_node_component, *output_sink);
    // }

  private:
    edge::IWritableAcceptorBase& get_output(SegmentAddress address) const override
    {
        return *this->get_source(address);
    }

    channel::Status process_one() override
    {
        input_message_t data;

        auto status = this->get_readable_edge()->await_read_for(data, channel::duration_t::zero());

        if (status == channel::Status::success)
        {
            // Use the tag to determine where it should go
            auto tag = data.first;

            this->get_writable_edge(tag)->await_write(std::move(data.second));
        }

        return status;
    }

    std::shared_ptr<node::LambdaNodeComponent<input_message_t, output_message_t>> m_node_component;
};

template <typename T>
class TypedManifold : public ManifoldBase
{
  public:
    TypedManifold(runnable::IRunnableResources& resources, std::string port_name) :
      ManifoldBase(resources,
                   std::move(port_name),
                   std::make_unique<ManifoldTagger<T>>(),
                   std::make_unique<ManifoldUnTagger<T>>())
    {}
};
}  // namespace mrc::manifold
