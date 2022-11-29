# Configuration Options

This example shows how altering two common configuration options (number of threads and channel size) of a MRC pipeline can change its behavior. By default MRC will set the thread count to match the number of cores in a system and will use a channel size of `128`. The channel size is expressed in number of elements regardless of the byte size of the objects.

In our previous examples the pipelines were quite simple. However in non-trivial pipelines it is quite likely that some nodes will execute faster than other nodes. When a reletively faster upstream source node emits data faster than they are able to be processed by a downstream sink node, it is possible that the channel will hit it's max channel size. When this happens the source node will block on the next write until there is room in the channel. Increasing the size of the channel would allow the source to emit as quickly as it is able to but at the cost of increased memory consumption.

In an ideal situation we have more cores and threads available than we have nodes in the pipeline, allowing for each node to run in their own thread without the need for a context switch. In more complex pipelines this may not always be be the case, and nodes will be scheduled as needed.

For this example the logging information is designed to show how messages move through the pipeline. For each of the 3 stages, they will output logs in the format:

```bash
${NODE_NAME}: ${LOG_MESSAGE} ${MESSAGE_ID}, TID: [${THREAD_NAME}]
```

Each message will generate exactly 3 messages as it moves through the pipeline

```bash
Source: Emitted    ${MESSAGE_ID}, TID: [${THREAD_NAME}]
...
Node  : Processing ${MESSAGE_ID}, TID: [${THREAD_NAME}]
...
Sink  : Got value  ${MESSAGE_ID}, TID: [${THREAD_NAME}]
```

### Note:
The threads will be created by MRC's underlying C++ implementation. Python's threading library will always return a thread name in the form of "Dummy-*" for any threads not created by Python.

In this example, it will be important to pay attention to the ordering of the messages, and the thread that they were executed on.

## Running the Example
This example will by default create a source which emits `10` integers, if we intentionally limit the channel size to 4 and limit our execution to a single thread the output should look like:

```bash
$ python ./docs/quickstart/python/mrc_qs_python/ex03_config_options/run.py --channel_size 4 --threads 1
mrc pipeline starting...
Source: Starting
Source: Emitted    00, TID: [Dummy-1]
Source: Emitted    01, TID: [Dummy-1]
Source: Emitted    02, TID: [Dummy-1]
Source: Emitted    03, TID: [Dummy-1]
Node  : Processing 00, TID: [Dummy-1]
Node  : Processing 01, TID: [Dummy-1]
Node  : Processing 02, TID: [Dummy-1]
Source: Emitted    04, TID: [Dummy-1]
Source: Emitted    05, TID: [Dummy-1]
Source: Emitted    06, TID: [Dummy-1]
Sink  : Got value  00, TID: [Dummy-1]
Sink  : Got value  01, TID: [Dummy-1]
Sink  : Got value  02, TID: [Dummy-1]
Node  : Processing 03, TID: [Dummy-1]
Node  : Processing 04, TID: [Dummy-1]
Node  : Processing 05, TID: [Dummy-1]
Source: Emitted    07, TID: [Dummy-1]
Source: Emitted    08, TID: [Dummy-1]
Source: Emitted    09, TID: [Dummy-1]
Source: Complete
Sink  : Got value  03, TID: [Dummy-1]
Sink  : Got value  04, TID: [Dummy-1]
Sink  : Got value  05, TID: [Dummy-1]
Node  : Processing 06, TID: [Dummy-1]
Node  : Processing 07, TID: [Dummy-1]
Node  : Processing 08, TID: [Dummy-1]
Sink  : Got value  06, TID: [Dummy-1]
Sink  : Got value  07, TID: [Dummy-1]
Sink  : Got value  08, TID: [Dummy-1]
Node  : Processing 09, TID: [Dummy-1]
Sink  : Got value  09, TID: [Dummy-1]
mrc pipeline complete.
```


If we up the channel size to 8 we get the following:

```bash
$ python ./docs/quickstart/python/mrc_qs_python/ex03_config_options/run.py --channel_size 8 --threads 1
mrc pipeline starting...
Source: Starting
Source: Emitted    00, TID: [Dummy-1]
Source: Emitted    01, TID: [Dummy-1]
Source: Emitted    02, TID: [Dummy-1]
Source: Emitted    03, TID: [Dummy-1]
Source: Emitted    04, TID: [Dummy-1]
Source: Emitted    05, TID: [Dummy-1]
Source: Emitted    06, TID: [Dummy-1]
Source: Emitted    07, TID: [Dummy-1]
Node  : Processing 00, TID: [Dummy-1]
Node  : Processing 01, TID: [Dummy-1]
Node  : Processing 02, TID: [Dummy-1]
Node  : Processing 03, TID: [Dummy-1]
Node  : Processing 04, TID: [Dummy-1]
Node  : Processing 05, TID: [Dummy-1]
Node  : Processing 06, TID: [Dummy-1]
Source: Emitted    08, TID: [Dummy-1]
Source: Emitted    09, TID: [Dummy-1]
Source: Complete
Sink  : Got value  00, TID: [Dummy-1]
Sink  : Got value  01, TID: [Dummy-1]
Sink  : Got value  02, TID: [Dummy-1]
Sink  : Got value  03, TID: [Dummy-1]
Sink  : Got value  04, TID: [Dummy-1]
Sink  : Got value  05, TID: [Dummy-1]
Sink  : Got value  06, TID: [Dummy-1]
Node  : Processing 07, TID: [Dummy-1]
Node  : Processing 08, TID: [Dummy-1]
Node  : Processing 09, TID: [Dummy-1]
Sink  : Got value  07, TID: [Dummy-1]
Sink  : Got value  08, TID: [Dummy-1]
Sink  : Got value  09, TID: [Dummy-1]
mrc pipeline complete.
```

If we add another thread:

```bash
$ python ./docs/quickstart/python/mrc_qs_python/ex03_config_options/run.py --channel_size 4 --threads 2
mrc pipeline starting...
Source: Starting
Source: Emitted    00, TID: [Dummy-1]
Source: Emitted    01, TID: [Dummy-1]
Source: Emitted    02, TID: [Dummy-1]
Source: Emitted    03, TID: [Dummy-1]
Node  : Processing 00, TID: [Dummy-2]
Source: Emitted    04, TID: [Dummy-1]
Node  : Processing 01, TID: [Dummy-2]
Source: Emitted    05, TID: [Dummy-1]
Node  : Processing 02, TID: [Dummy-2]
Source: Emitted    06, TID: [Dummy-1]
Node  : Processing 03, TID: [Dummy-2]
Source: Emitted    07, TID: [Dummy-1]
Sink  : Got value  00, TID: [Dummy-2]
Sink  : Got value  01, TID: [Dummy-2]
Sink  : Got value  02, TID: [Dummy-2]
Node  : Processing 04, TID: [Dummy-2]
Source: Emitted    08, TID: [Dummy-1]
Node  : Processing 05, TID: [Dummy-2]
Source: Emitted    09, TID: [Dummy-1]
Node  : Processing 06, TID: [Dummy-2]
Source: Complete
Sink  : Got value  03, TID: [Dummy-2]
Sink  : Got value  04, TID: [Dummy-2]
Sink  : Got value  05, TID: [Dummy-2]
Node  : Processing 07, TID: [Dummy-2]
Node  : Processing 08, TID: [Dummy-2]
Node  : Processing 09, TID: [Dummy-2]
Sink  : Got value  06, TID: [Dummy-2]
Sink  : Got value  07, TID: [Dummy-2]
Sink  : Got value  08, TID: [Dummy-2]
Sink  : Got value  09, TID: [Dummy-2]
mrc pipeline complete.
```

And 3 threads:

```bash
$ python ./docs/quickstart/python/mrc_qs_python/ex03_config_options/run.py --channel_size 4 --threads 3
mrc pipeline starting...
Source: Starting
Source: Emitted    00, TID: [Dummy-1]
Source: Emitted    01, TID: [Dummy-1]
Source: Emitted    02, TID: [Dummy-1]
Source: Emitted    03, TID: [Dummy-1]
Node  : Processing 00, TID: [Dummy-2]
Source: Emitted    04, TID: [Dummy-1]
Node  : Processing 01, TID: [Dummy-2]
Source: Emitted    05, TID: [Dummy-1]
Node  : Processing 02, TID: [Dummy-2]
Sink  : Got value  00, TID: [Dummy-3]
Source: Emitted    06, TID: [Dummy-1]
Node  : Processing 03, TID: [Dummy-2]
Source: Emitted    07, TID: [Dummy-1]
Node  : Processing 04, TID: [Dummy-2]
Sink  : Got value  01, TID: [Dummy-3]
Source: Emitted    08, TID: [Dummy-1]
Node  : Processing 05, TID: [Dummy-2]
Sink  : Got value  02, TID: [Dummy-3]
Sink  : Got value  03, TID: [Dummy-3]
Node  : Processing 06, TID: [Dummy-2]
Sink  : Got value  04, TID: [Dummy-3]
Node  : Processing 07, TID: [Dummy-2]
Source: Emitted    09, TID: [Dummy-1]
Sink  : Got value  05, TID: [Dummy-3]
Sink  : Got value  06, TID: [Dummy-3]
Node  : Processing 08, TID: [Dummy-2]
Sink  : Got value  07, TID: [Dummy-3]
Sink  : Got value  08, TID: [Dummy-3]
Source: Complete
Node  : Processing 09, TID: [Dummy-2]
Sink  : Got value  09, TID: [Dummy-3]
mrc pipeline complete.
```
