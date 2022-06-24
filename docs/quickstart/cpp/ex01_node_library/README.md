# Node Library

This example illustrates how to create a library of reusable SRF nodes that can be later used to form pipelines. Building a library of reusable nodes can facilitate rapid pipeline development but it requires using a different interface when creating the nodes from the `make_source`, `make_node`, and `make_sink` methods shown in Example 1.

In this example, we will only be creating a single source node that functions identically to the source node in Example 1. If you recall, we created that node using the following code:

```cpp
auto source = s.make_source<int>("int_source", [](rxcpp::subscriber<int> s) {
    s.on_next(1);
    s.on_next(2);
    s.on_next(3);
    s.on_completed();
});
```

While this syntax makes it quick to define a one-off node, it can become very verbose when the same node is needed in multiple places. To create a reusable node, we need to inherit from one of the following classes:

- Source: Inherit from `srf::node::RxSource<T>`
- Node: Inherit from `srf::node::RxNode<T, R>`
- Sink: Inherit from `srf::node::RxSink<R>`

## Creating the Source

Converting the above source that uses `make_source` into a class that derives from `RxSource` looks like the following:

```cpp
class IntSource : public srf::node::RxSource<int>
{
  public:
    IntSource() :
      srf::node::RxSource<int>(rxcpp::observable<>::create<int>([](rxcpp::subscriber<int> s) {
          s.on_next(1);
          s.on_next(2);
          s.on_next(3);
          s.on_completed();
      }))
    {}
};
```

We can see that the constructor for `RxSource` takes the same argument that was passed to `make_source`. The main difference here is that we can now create multiple copies of `IntSource` without needing to redefine the same lambda each time.

## Building the Library

To add this node into a C++ library, we need to create the following CMakeLists.txt file:

```cmake
add_library(ex01_node_library SHARED
  src/nodes.cpp
)

target_include_directories(ex01_node_library
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(ex01_node_library
  PUBLIC
    srf::libsrf
)
```

As far as CMake libraries go, there is nothing out of the ordinary that needs to be done to build the component into a library. Simply create a library as usual and be sure to add `srf::libsrf` to the library list in `target_link_libraries`.

## Running the Example

In this example, we have only created a reusable library. To see an executable that uses this library, go to the next example, [`ex02_pipeline_with_library`](../ex02_pipeline_with_librar)
