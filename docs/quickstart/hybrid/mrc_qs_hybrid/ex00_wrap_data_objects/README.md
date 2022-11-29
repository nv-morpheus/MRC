# Simple Pipeline

This example illustrates how to wrap C++ objects in python and then pass these C++ objects between python nodes in a pipeline.

## Running the Example

We can see this simple pipeline in action by running the python script:

```bash
$ python ./docs/quickstart/hybrid/mrc_qs_hybrid/ex00_wrap_data_objects/run.py
I20220617 00:29:21.517798 32164 run.py:75] mrc pipeline starting...
I20220617 00:29:21.519217 32171 run.py:30] Processing 'Instance-0'
I20220617 00:29:21.519305 32171 run.py:30] Processing 'Instance-1'
I20220617 00:29:21.519374 32171 run.py:30] Processing 'Instance-2'
I20220617 00:29:21.519484 32171 run.py:46] Got value: {Name: 'Instance-0', Value: 0}, Incrementing counter
I20220617 00:29:21.519575 32171 run.py:46] Got value: {Name: 'Instance-1', Value: 2}, Incrementing counter
I20220617 00:29:21.519649 32171 run.py:46] Got value: {Name: 'Instance-2', Value: 4}, Incrementing counter
I20220617 00:29:21.519997 32164 run.py:83] mrc pipeline complete: total_sum should be 6; total_sum=6
```
