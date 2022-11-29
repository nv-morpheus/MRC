# Custom Data

Building pipelines with only integers and floats doesn't provide much utility. Instead, we need to be able to pass any type of object between nodes and modify it at each point. In this example, we demonstrate how a custom class can be defined and used as the object passed between nodes.

Before defining our pipeline, we have created a simple `dataclass` class called `MyCustomClass`:

```python
@dataclasses.dataclass
class MyCustomClass:
    """
    This is our custom data class that is used to store both a name and a value.
    """

    value: int
    name: str
```

This class is pretty trivial, but it shows how its possible to pass any object between nodes. We are going to replicate a similar pipeline to `00_SimplePipeline` except we will use our new class. In addition to passing a value between the stages, we will also pass a name to make it easier to see the original value.

Looking at the new node processing function, you can see that we will print the class' name property at the time its value is modified:

```python
def update_obj(x: MyCustomClass):

    print("Processing '{}'".format(x.name))

    # Alter the value property of the class
    x.value = x.value * 2

    return x
```

Unlike the previous example where we count the number of items emitted, we instead will sum the total of the `value` property in the sink:

```python
def sink_on_next(x: MyCustomClass):

    nonlocal total_sum
    total_sum += x.value
```

## Running the Example

Running the example yields the following:

```bash
$ python ./docs/quickstart/python/mrc_qs_python/ex01_custom_data/run.py
mrc pipeline starting...
Processing 'Instance-0'
Processing 'Instance-1'
Processing 'Instance-2'
mrc pipeline complete: total_sum should be 6; total_sum=6
```

Looking at the output, we can see that 3 objects were procesed by this pipeline, one for each object emmitted by the source. Unlike previous examples, we printed the names of each object in the interior node, instead of in the sink.
