import argparse
import os

from pip._internal.commands.show import search_packages_info


def gen_dep_file(pkg_name: str, input_file: str, output_file: str):

    package_generator = search_packages_info([pkg_name])

    for pkg_info in package_generator:

        if (pkg_info.name == pkg_name):

            joined_files = " ".join([os.path.join(pkg_info.location, f) for f in pkg_info.files])

            # Create the output lines
            lines = [f"{input_file}: {joined_files}"]

            # Write the depfile
            with open(output_file, "w") as f:
                f.writelines(lines)

            break


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("--pkg_name", type=str, required=True, help='an integer for the accumulator')
    parser.add_argument('--input_file', type=str, required=True, help='sum the integers (default: find the max)')
    parser.add_argument('--output_file', type=str, help='sum the integers (default: find the max)')

    args = parser.parse_args()

    if (args.output_file is None):
        args.output_file = args.input_file + ".d"

    gen_dep_file(pkg_name=args.pkg_name, input_file=args.input_file, output_file=args.output_file)
