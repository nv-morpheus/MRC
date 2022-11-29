# Mixed Execution

This example illustrates how to run a pipeline with nodes that are built in both C++ and python. The overall pipeline is very similar to the previous example, `ex01_wrap_nodes`, except we have added a runtime option to toggle whether the source, node or sink is built using python or C++. For each of the source, node and sink, there is code similar to the following:

```python
# Create the source depending on the runtime option
if (python_source):
    src = seg.make_source("source", source_gen())
else:
    src = DataObjectSource(seg, "src", count=3)
```

Depending on the value of `python_source`, will control what type of source node is created. The implementations of our python and C++ sources are very similar, except the python version will append `[Python]` to the instance name.

## Running the Example

We can see this simple pipeline in action by running the python script:

```bash
$ python ./docs/quickstart/hybrid/mrc_qs_hybrid/ex02_mixed_execution/run.py
I20220617 00:19:01.001237 30754 run.py:83] mrc pipeline starting...
I20220617 00:19:01.002104 30761 nodes.cpp:102] my_seg/sink; rank: 0; size: 1; tid: 140073000851200; fid: 0x7f65280a3d00 Got value: {Name: 'Instance-0', Value: 0}
I20220617 00:19:01.002135 30761 nodes.cpp:102] my_seg/sink; rank: 0; size: 1; tid: 140073000851200; fid: 0x7f65280a3d00 Got value: {Name: 'Instance-1', Value: 2}
I20220617 00:19:01.002148 30761 nodes.cpp:102] my_seg/sink; rank: 0; size: 1; tid: 140073000851200; fid: 0x7f65280a3d00 Got value: {Name: 'Instance-2', Value: 4}
I20220617 00:19:01.002161 30761 nodes.cpp:110] my_seg/sink; rank: 0; size: 1; tid: 140073000851200; fid: 0x7f65280a3d00 Completed
I20220617 00:19:01.002429 30754 run.py:91] mrc pipeline complete
```

By default, the pipeline will use all C++ nodes and messages will be passed between the nodes without ever touching the python runtime. This is especially important for performance because it avoids competition on the GIL. If we add the flag `--python_source` we can see the output change slightly:

```bash
$ python ./docs/quickstart/hybrid/mrc_qs_hybrid/ex02_mixed_execution/run.py --python_source
I20220617 00:21:07.025996 31038 run.py:83] mrc pipeline starting...
I20220617 00:21:07.026932 31045 nodes.cpp:102] my_seg/sink; rank: 0; size: 1; tid: 140306719217408; fid: 0x7f9b9c0a3e00 Got value: {Name: '[Python]Instance-0', Value: 0}
I20220617 00:21:07.026962 31045 nodes.cpp:102] my_seg/sink; rank: 0; size: 1; tid: 140306719217408; fid: 0x7f9b9c0a3e00 Got value: {Name: '[Python]Instance-1', Value: 2}
I20220617 00:21:07.026975 31045 nodes.cpp:102] my_seg/sink; rank: 0; size: 1; tid: 140306719217408; fid: 0x7f9b9c0a3e00 Got value: {Name: '[Python]Instance-2', Value: 4}
I20220617 00:21:07.026988 31045 nodes.cpp:110] my_seg/sink; rank: 0; size: 1; tid: 140306719217408; fid: 0x7f9b9c0a3e00 Completed
I20220617 00:21:07.027138 31038 run.py:91] mrc pipeline complete
```

This might not seem very interesting. The only thing that changed is the `Name` property now looks like `[Python]Instance-0` instead of `Instance-0`. But what happens if you have a difficult to diagnose bug that would be much easier to solve if you could inspect the messages in python between the source and sink? If we use the `--python_node` argument, we could do exactly that:

```bash
$ python ./docs/quickstart/hybrid/mrc_qs_hybrid/ex02_mixed_execution/run.py --python_node
I20220617 00:23:38.774160 31348 run.py:83] mrc pipeline starting...
I20220617 00:23:38.776108 31355 run.py:36] [Python] Processing 'Instance-0'
I20220617 00:23:38.776352 31355 run.py:36] [Python] Processing 'Instance-1'
I20220617 00:23:38.776551 31355 run.py:36] [Python] Processing 'Instance-2'
I20220617 00:23:38.776713 31355 nodes.cpp:102] my_seg/sink; rank: 0; size: 1; tid: 140483438864128; fid: 0x7fc4b80a5500 Got value: {Name: 'Instance-0', Value: 0}
I20220617 00:23:38.776772 31355 nodes.cpp:102] my_seg/sink; rank: 0; size: 1; tid: 140483438864128; fid: 0x7fc4b80a5500 Got value: {Name: 'Instance-1', Value: 2}
I20220617 00:23:38.776803 31355 nodes.cpp:102] my_seg/sink; rank: 0; size: 1; tid: 140483438864128; fid: 0x7fc4b80a5500 Got value: {Name: 'Instance-2', Value: 4}
I20220617 00:23:38.776850 31355 nodes.cpp:110] my_seg/sink; rank: 0; size: 1; tid: 140483438864128; fid: 0x7fc4b80a5500 Completed
I20220617 00:23:38.777143 31348 run.py:91] mrc pipeline complete
```

Now, we have converted our pipeline to run C++ -> Python -> C++. The interior node is executing the following function to generate the additional logs:

```python
def update_obj(x: DataObject):

    logger.info("[Python] Processing '{}'".format(x.name))

    # Alter the value property of the class
    x.value = x.value * 2

    return x
```

Which is why we can now see the additional `[Python] Processing` messages. If you ran this example in an IDE that is capable of putting breakpoints in python code, you could stop the pipeline execution and debug messages as they are sent. Once you figure out the issue, you can revert back to the C++ stages for performance.

What are the downsides to having a mixed execution pipeline? What are the limitations?
