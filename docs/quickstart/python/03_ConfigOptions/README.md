# Configuration Options

This example shows how altering the configuration options of a SRF pipeline can change its behavior. Again, we are using a very simple pipeline with a single source, node and sink. However, for this example, the logging information is designed to show how messages move through the pipeline. For each of the 3 stages, they will output logs in the format:

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

In this example, it will be important to pay attention to the ordering of the messages, and the thread that they were executed on.

## Running the Example

If we run this example with the default options, we will get the following output:

```bash
$ python ./docs/quickstart/python/03_ConfigOptions/run.py
srf pipeline starting...
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
srf pipeline complete.
```

By default, we run with 1 thread and a buffer of size 4 between the nodes.

If we up the buffer size to 8 we get the following:

```bash
$ python ./docs/quickstart/python/03_ConfigOptions/run.py --channel_size 8
srf pipeline starting...
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
srf pipeline complete.
```

If we add another thread:

```bash
$ python ./docs/quickstart/python/03_ConfigOptions/run.py --threads 2
srf pipeline starting...
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
srf pipeline complete.
```

And 3 threads:

```bash
$ python ./docs/quickstart/python/03_ConfigOptions/run.py --threads 3
srf pipeline starting...
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
srf pipeline complete.
```
