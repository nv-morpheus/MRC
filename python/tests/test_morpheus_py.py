# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from io import StringIO

import numpy as np
import nvtx
import pytest
from morpheus.pipeline.input.from_file import df_onread_cleanup
from morpheus.pipeline.messages import MultiInferenceMessage
from morpheus.pipeline.preprocessing import DeserializeStage
from morpheus.pipeline.preprocessing import PreprocessNLPStage
from tqdm import tqdm

import cudf

import srf
import srf.core.morpheus as srfm

global progress

# Set environment var MORPHEUS_ROOT to the location of a morpheus repo root. Need the data files
MORPHEUS_ROOT = os.environ.get("MORPHEUS_ROOT", default="")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MORPHEUS_ROOT),
    reason="'MORPHEUS_ROOT' environment variable not found. Cannot load data files. Skipping module.")


def _generate_frames(df: cudf.DataFrame, batch_size: int, iterative: bool = True):
    count = 0
    out = []

    for x in df.groupby(np.arange(len(df)) // batch_size):
        y = x[1].reset_index(drop=True)

        count += 1

        if (iterative):
            yield y
        else:
            out.append(y)

    if (not iterative):
        yield out


def segment_init(seg: srf.Builder):
    def from_file_fn(s: srf.Subscriber):

        input_file = os.path.join(MORPHEUS_ROOT, "data/pcap_dump.jsonlines")

        with open(input_file) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]

        lines_buffer = StringIO("\n".join(lines))

        df = cudf.read_json(lines_buffer, engine="cudf", lines=True)

        df = df_onread_cleanup(df)

        # Save the original
        df["_orig"] = lines

        repeat = 1

        for _ in range(repeat):
            for x in _generate_frames(df, 1024, True):
                # print("Calling on_next from from-file")
                with nvtx.annotate("File::on_next", color="green", domain="python"):
                    s.on_next(x)

        s.on_completed()

    # df = cudf.read_json(input_file, engine="cudf", lines=True)

    # srfm.test_cudf(df["data"]._column)

    from_file = seg.make_source("from-file", from_file_fn)
    # from_file = srfm.make_file_source(seg, "from-file", input_file)

    # Deserialize stage
    @nvtx.annotate("Deserialize", color="red", domain="python")
    def deserialize_fn(x: cudf.DataFrame):
        return DeserializeStage.process_dataframe(x)

    deserialize = seg.make_node("deserialize", deserialize_fn)

    seg.make_edge(from_file, deserialize)

    # Preprocess
    @nvtx.annotate("Preprocess", color="blue", domain="python")
    def preprocess_fn(x: srfm.MultiMessage):
        try:
            return PreprocessNLPStage.pre_process_batch(x,
                                                        vocab_hash_file=os.path.join(
                                                            MORPHEUS_ROOT, "data/bert-base-cased-hash.txt"),
                                                        do_lower_case=False,
                                                        seq_len=256,
                                                        stride=192,
                                                        truncation=True,
                                                        add_special_tokens=False)
        except Exception as ex:
            print(ex)

    preprocess = seg.make_node("preprocess", preprocess_fn)
    # preprocess = srfm.make_preprocessor(seg, "preprocess")

    seg.make_edge(deserialize, preprocess)

    # Preprocess
    def inference_fn(input: srf.Observable, output: srf.Subscriber):
        def obs_on_next(x: MultiInferenceMessage):
            # @nvtx.annotate("Inference", color="yellow", domain="python")
            output.on_next(x)

        def obs_on_error(x):
            output.on_error(x)

        def obs_on_completed():
            output.on_completed()

        obs = srf.Observer.make_observer(obs_on_next, obs_on_error, obs_on_completed)

        input.subscribe(obs)

    inference = seg.make_node_full("inference", inference_fn)

    seg.make_edge(preprocess, inference)

    def sink_on_next(x):
        progress.update(n=x.count)

    def sink_on_error(x):
        print("Got error: {}".format(x))

    def sink_on_completed():
        print("Got completed")

    sink = seg.make_sink("my_sink", sink_on_next, sink_on_error, sink_on_completed)

    seg.make_edge(inference, sink)
    # srfm.make_cxx2py_edge2(seg, preprocess, sink)


def test_morpheus():

    pipeline = srf.Pipeline()

    global progress
    progress = tqdm(desc="Message Rate",
                    smoothing=0.001,
                    dynamic_ncols=True,
                    unit="message",
                    mininterval=0.25,
                    maxinterval=1.0)

    pipeline.make_segment("my_seg", segment_init)

    options = srf.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-2"

    executor = srf.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()


if (__name__ == "__main__"):
    test_morpheus()
