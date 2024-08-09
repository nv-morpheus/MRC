import time

import cupy as cp
import numpy as np
import ray
from ray.dag import InputNode
from ray.dag import MultiOutputNode

import mrc
import mrc.core.operators as ops

ray.init(num_cpus=2)

num_msg = 10000

large_object = np.zeros((100 * 1024) // 8)


class TrackTimings:

    def __init__(self, id, payload):
        self.id = id
        self.timings = []
        self.payload = payload

        self.add_timing()

    def add_timing(self):
        self.timings.append(time.time())

    def calc_latency(self):

        # Calculate the latency between each step
        latencies = []

        for i in range(1, len(self.timings)):
            latencies.append(self.timings[i] - self.timings[i - 1])

        return latencies

    def avg_latency(self):
        latencies = self.calc_latency()
        return sum(latencies) / len(latencies)

    def print_info(self):
        print(f"Latencies: {self.calc_latency()}")
        print(f"Avg Latency: {self.avg_latency() * 1000} ms")
        print(f"Total Latency: {(self.timings[-1] - self.timings[0]) * 1000} ms")


@ray.remote
class Step:

    def step(self, x: TrackTimings):

        # y = len(large_object)

        x.add_timing()

        return x


step = Step.remote()

with InputNode() as dag_input:
    a_ref = step.step.bind(dag_input)
    b_ref = step.step.bind(a_ref)
    c_ref = step.step.bind(b_ref)
    d_ref = step.step.bind(b_ref)

    dag = d_ref

# dag = dag.experimental_compile()

print("==============Starting Ray DAG==============")
ray_start = time.time()

timings = ray.get([dag.execute(TrackTimings(i, np.zeros((0 * 1024 * 1024) // 8))) for i in range(num_msg)])

for timing in timings:
    if (timing.id == num_msg - 1):
        timing.print_info()

ray_end = time.time()
print(f"Total Time: {(ray_end - ray_start) * 1000} ms")
print("==============Ending Ray DAG==============")


def init_segment(builder: mrc.Builder):

    def source_fn():
        for i in range(num_msg):
            # time.sleep(0.1)
            yield TrackTimings(i, np.zeros((0 * 1024 * 1024) // 8))

    def node_fn(message: TrackTimings):
        message.add_timing()
        return message

    def sink_fn(message: TrackTimings):
        message.add_timing()

        if (message.id == num_msg - 1):
            message.print_info()
        # message.print_info()

    source = builder.make_source("source", source_fn)

    node_a = builder.make_node("a", ops.map(node_fn))
    node_b = builder.make_node("b", ops.map(node_fn))

    sink = builder.make_sink("sink", sink_fn)

    builder.make_edge(source, node_a)
    builder.make_edge(node_a, node_b)
    builder.make_edge(node_b, sink)


pipe = mrc.Pipeline()

pipe.make_segment("TestSegment11", init_segment)

options = mrc.Options()
options.topology.user_cpuset = "0"

executor = mrc.Executor(options)
executor.register_pipeline(pipe)

print("==============Starting MRC==============")
mrc_start = time.time()

executor.start()
executor.join()

mrc_end = time.time()
print(f"Total Time: {(mrc_end - mrc_start) * 1000} ms")
print("==============Ending MRC==============")
