import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


root = Path(__file__).resolve().parents[1]


def run(command):
    result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError("Command failed: " + shlex.join(map(str, command)))
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Compile the CUDA examples and check GPU results.")
    parser.add_argument("--compile-only", action="store_true", help="Do not create a CUDA context or run GPU tests.")
    parser.add_argument("--arch", help="Optional nvcc target, for example sm_80 to enable async copies.")
    args = parser.parse_args()
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required and must be on PATH")
    flags = [nvcc, "-std=c++17", "-Xcompiler=-Wall,-Wextra"]
    if args.arch:
        flags.append("-arch=" + args.arch)

    for filename in [root / "check.py", Path(__file__)]:
        compile(filename.read_text(), str(filename), "exec")
    print("PASS Python syntax", flush=True)

    opencv_flags = None
    if shutil.which("pkg-config"):
        result = subprocess.run(["pkg-config", "--cflags", "--libs", "opencv4"], text=True, capture_output=True)
        if result.returncode == 0:
            opencv_flags = shlex.split(result.stdout)

    # Build in a temporary directory so existing user binaries are never replaced.
    with tempfile.TemporaryDirectory(prefix="cuda-checks-") as directory:
        build = Path(directory)
        jobs = []
        executables = []
        image_sources = {"Blur.cu", "GaussBlur.cu"}
        for source in sorted(root.glob("*.cu")):
            output = build / source.stem
            command = flags + [str(source), "-o", str(output)]
            if source.name in image_sources:
                if opencv_flags is None:
                    command = flags + ["-DCUDA_KERNEL_TEST", "-c", str(source), "-o", str(output) + ".o"]
                else:
                    command += opencv_flags
            else:
                executables.append(output)
            jobs.append((source.name, command))

        # Compile the separate blur kernel even when OpenCV is unavailable.
        jobs.append(("header_file/GBlur.cu", flags + ["-c", str(root / "header_file/GBlur.cu"), "-o", str(build / "GBlur.o")]))
        if opencv_flags is not None:
            jobs.append(("header_file/main.cu", flags + [str(root / "header_file/main.cu"), str(root / "header_file/GBlur.cu"),
                                                          "-o", str(build / "GBlur")] + opencv_flags))
        else:
            print("SKIP OpenCV host programs: opencv4 is missing; blur kernels are still compiled and included in GPU tests.", flush=True)

        regression = build / "verify_kernels"
        jobs.append(("kernel regression tests", flags + [str(root / "tests/verify_kernels.cu"), "-o", str(regression)]))
        # Independently compile the asynchronous branch even with nvcc's older default target.
        jobs.append(("FlashAttention pipeline sm_80", [nvcc, "-std=c++17", "-arch=sm_80", "-c",
                                                       str(root / "flash_attention_3.cu"), "-o", str(build / "flash3-sm80.o")]))
        # SourceModule normally needs a GPU context; its literal CUDA code can be compiled offline.
        module = ast.parse((root / "check.py").read_text())
        source_strings = [node.args[0].value for node in ast.walk(module)
                          if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                          and node.func.id == "SourceModule" and node.args
                          and isinstance(node.args[0], ast.Constant)]
        for i, source_string in enumerate(source_strings):
            source = build / ("pycuda-source-" + str(i) + ".cu")
            source.write_text(source_string)
            jobs.append(("PyCUDA SourceModule", flags + ["-c", str(source), "-o", str(source) + ".o"]))

        def compile_one(job):
            name, command = job
            run(command)
            print("PASS compile " + name, flush=True)

        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(compile_one, jobs))
        print("Compiled " + str(len(jobs)) + " targets.", flush=True)
        if args.compile_only:
            print("GPU runtime checks were not run (--compile-only).")
            return

        print(run([str(regression)]), end="", flush=True)
        for executable in executables:
            run([str(executable)])
            print("PASS run " + executable.name, flush=True)
        try:
            import pycuda
        except ImportError:
            print("SKIP PyCUDA runtime: install pycuda and numpy to run check.py")
        else:
            run([sys.executable, str(root / "check.py")])
            print("PASS run check.py", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
