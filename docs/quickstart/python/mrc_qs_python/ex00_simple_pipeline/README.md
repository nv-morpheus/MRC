# Simple Pipeline

This example illustrates how to create a simple pipeline with a single source, node, and sink connected together. Each of the nodes in the segment ("node" here refers to either a source, sink or an object that is both a source and sink) is responsible for a simple task:

- Source: Creates 3 integers in the sequence 1, 2, 3
- Node: Transforms the integer by multiplying it by 2.5, resulting in a float
- Sink: Prints any received value and sums the total number of items

Each of the objects in the Segment is created using the `Segment.make_XXX(NAME, ...)` function where `XXX` is replace with either `source`, `sink` or `node`.

Once each object is created, they can be linked together using `Segment.make_edge(SOURCE, SINK)`. There are a few rules when making edges:

- Source objects can only appear in the left-hand argument
- Sink objects can only appear in the right-hand argument
- Node objects can appear in either side
- Sources can only be connected to one downstream sink
  - To use multiple downstream sinks with broadcast or round-robin functionality, see the guide on operators
- Sinks can accept multiple upstream sources
  - Its possible to connect two sources to a single sink. The sink will process the messages in the order that they are received in

## Running the Example

We can see this simple pipeline in action by running the python script:

```bash
$ python docs/quickstart/python/mrc_qs_python/ex00_simple_pipeline/run.py
mrc pipeline starting...
Got value: 2.5, Incrementing counter
Got value: 5.0, Incrementing counter
Got value: 7.5, Incrementing counter
mrc pipeline complete: counter should be 3; counter=3
```

We can see that the sink function was called 3 times, one for each value emitted by the source. What happens if you change the number of `yield` statements in the source object?
