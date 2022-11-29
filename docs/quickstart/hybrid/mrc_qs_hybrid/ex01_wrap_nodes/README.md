# Wrap Nodes

In this example, we illustrate how to make python wrappers around C++ nodes to allow for a C++ pipeline configurations from python.
## Running the Example

We can see this simple pipeline in action by running the python script:

```bash
$ python ./docs/quickstart/hybrid/mrc_qs_hybrid/ex01_wrap_nodes/run.py
I20220617 00:28:01.370167 32005 run.py:46] mrc pipeline starting...
I20220617 00:28:01.371134 32012 nodes.cpp:102] my_seg/sink; rank: 0; size: 1; tid: 139784256304896; fid: 0x7f22000a3d00 Got value: {Name: 'Instance-0', Value: 0}
I20220617 00:28:01.371173 32012 nodes.cpp:102] my_seg/sink; rank: 0; size: 1; tid: 139784256304896; fid: 0x7f22000a3d00 Got value: {Name: 'Instance-1', Value: 2}
I20220617 00:28:01.371197 32012 nodes.cpp:102] my_seg/sink; rank: 0; size: 1; tid: 139784256304896; fid: 0x7f22000a3d00 Got value: {Name: 'Instance-2', Value: 4}
I20220617 00:28:01.371218 32012 nodes.cpp:110] my_seg/sink; rank: 0; size: 1; tid: 139784256304896; fid: 0x7f22000a3d00 Completed
I20220617 00:28:01.371505 32005 run.py:54] mrc pipeline complete
```
