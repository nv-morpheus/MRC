# -*- python -*-
import os
import subprocess
import sys

import gdb

conda_env_path = os.environ.get("CONDA_PREFIX", None)

if (conda_env_path is not None):

    gcc_path = os.environ.get("GCC", None)

    if (gcc_path is None):
        print(
            "Could not find gcc from $GCC: '{}'. Ensure gxx_linux-64, gcc_linux-64, sysroot_linux-64, and gdb have been installed into the current conda environment"
            .format(gcc_path))
    else:
        # Get the GCC version
        result = subprocess.run([gcc_path, '-dumpversion'], stdout=subprocess.PIPE)
        gcc_version = result.stdout.decode("utf-8").strip()

        # Build the gcc python path
        gcc_python_path = os.path.join(conda_env_path, "share", "gcc-{}".format(gcc_version), "python")

        if (os.path.exists(gcc_python_path)):

            # Add to the path for the pretty printer
            sys.path.insert(0, gcc_python_path)

            # Now register the pretty printers
            from libstdcxx.v6 import register_libstdcxx_printers
            register_libstdcxx_printers(gdb.current_objfile())

            print("Loaded stdlibc++ pretty printers")
        else:
            print("Could not find gcc python files at: {}".format(gcc_python_path))
            print(
                "Ensure gxx_linux-64, gcc_linux-64, sysroot_linux-64, and gdb have been installed into the current conda environment"
            )
