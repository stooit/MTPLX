#!/usr/bin/env python3
"""Package the tested QSA extension inside a platform-specific MTPLX wheel.

The pure Python wheel remains the fallback for other Python/OS platforms.
This builds a local artifact only; it never uploads or installs anything.
"""

import argparse
import hashlib
import json
from pathlib import Path

from packaging.tags import Tag
from packaging.utils import parse_wheel_filename
from wheel.wheelfile import WheelFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    name, version, _, core_tags = parse_wheel_filename(args.runtime.name)
    native_name, _, _, native_tags = parse_wheel_filename(args.native.name)
    if name != "mtplx" or core_tags != frozenset({Tag("py3", "none", "any")}):
        parser.error("The runtime input must be the pure Python MTPLX wheel")
    if native_name.replace("-", "_") != "mtplx_qsa_kernels" or len(native_tags) != 1:
        parser.error("Expected one tested mtplx_qsa_kernels platform wheel")
    tag = next(iter(native_tags))
    if not tag.platform.startswith("macosx_") or not tag.platform.endswith("_arm64"):
        parser.error("The native wheel must target Apple Silicon")
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / f"mtplx-{version}-{tag}.whl"
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    with WheelFile(args.runtime) as core, WheelFile(args.native) as native:
        members = [n for n in native.namelist() if n.startswith("mtplx_qsa_kernels/")]
        for required in ("NOTICE", "LICENSE.txt", "MLX_LICENSE.txt"):
            if f"mtplx_qsa_kernels/{required}" not in members:
                parser.error(f"Native attribution is missing: {required}")
        if not any(n.endswith(".metallib") for n in members) or not any(n.endswith(".so") for n in members):
            parser.error("Native wheel lacks its extension or Metal library")
        provenance = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (args.runtime, args.native)}
        with WheelFile(output, "w") as bundled:
            for archive, names in ((core, core.namelist()), (native, members)):
                for name in names:
                    if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                        continue
                    if name.startswith("/") or ".." in Path(name).parts:
                        raise ValueError(f"Unsafe archive path: {name}")
                    data = archive.read(name)  # WheelFile verifies source RECORD hashes.
                    if archive is core and name.endswith(".dist-info/WHEEL"):
                        lines = [line for line in data.decode().splitlines()
                                 if not line.startswith(("Tag:", "Root-Is-Purelib:"))]
                        data = ("\n".join([*lines, "Root-Is-Purelib: false", f"Tag: {tag}", ""])).encode()
                    elif archive is core and name.endswith(".dist-info/top_level.txt"):
                        data += b"mtplx_qsa_kernels\n"
                    bundled.writestr(archive.getinfo(name), data)
            bundled.writestr("mtplx/native_build_receipt.json", json.dumps(provenance, indent=2))
            # WheelFile writes a new RECORD for the complete distribution.
    print(output)


if __name__ == "__main__":
    main()
