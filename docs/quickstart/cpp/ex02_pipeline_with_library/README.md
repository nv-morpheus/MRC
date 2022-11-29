# Pipeline with Library

This example illustrates how to use the node library that was created in `ex01_node_library`. Following the pipeline in `ex00_simple_pipeline`, we will simply replace the source with the following:

```cpp
auto source = s.construct_object<IntSource>("int_source");
```

Where `IntSource` is the compiled class from the previous example. The most important thing to note here is the call to `construct_object<NodeT>(std::string name, ...args)`, where `NodeT` is the type of node to create, `name` is the node name, and `args` is a variable number of arguments. To ensure that each node can store the necessary information to function in the pipeline, we wrap all nodes in a `mrc::segment::Object` class. This alleviates the need to define properties like `name` on each and every node class. The downside here is that nodes should not be created directly, but instantiated using the `construct_object` function.

The `construct_object` function's first argument is the name to assign to the node. The same name that is used in `make_source`. The remainder of the arguments are passed directly into the node's constructor. So calling the following function:

```cpp
auto source = s.construct_object<MySource>("source_name", 1, 2.5f, "string_argument");
```

Would be equivalent to:

```cpp
auto source = std::make_shared<MySource>(1, 2.5f, "string_argument");
```

After the node is created using `construct_object`, the rest of the pipeline is constructed exactly the same as it was in `ex00_simple_pipeline`.

## Running the Example

Exactly the same as `ex00_simple_pipeline`, we can run the pipeline and get the following output:

```bash
$ ${QSG_BUILD_DIR}/docs/quickstart/cpp/ex02_pipeline_with_library/ex02_pipeline_with_library.x
mrc pipeline starting...
sink: 2.5
sink: 5
sink: 7.5
mrc pipeline complete: counter should be 3; counter=3
```

Where `${QSG_BUILD_DIR}` is the output location of the CMake build.
