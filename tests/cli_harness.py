import subprocess
import os
import sys

bin_path = os.path.abspath(r"target\release\luraph_deobf_rs.exe")
if not os.path.exists(bin_path):
    bin_path = os.path.abspath(r"target\debug\luraph_deobf_rs.exe")

print(f"Target binary: {bin_path}")

tests = [
    ("No arguments", []),
    ("Missing -o argument", ["-i", "non_existent.lua"]),
    ("Missing -i argument", ["-o", "out.lua"]),
    ("Unknown flag", ["-i", "in.lua", "-o", "out.lua", "--unknown-flag-xyz"]),
    ("Non-existent input file", ["-i", "nonexistent_payload_12345.lua", "-o", "out.lua"]),
    ("Directory as input", ["-i", "src", "-o", "out.lua"]),
    ("Zero threads (-t 0)", ["-i", "Cargo.toml", "-o", "test_out_0.lua", "-t", "0"]),
    ("Negative threads (-t -1)", ["-i", "Cargo.toml", "-o", "test_out_neg.lua", "-t", "-1"]),
    ("Extreme memory limit zero (-m 0)", ["-i", "Cargo.toml", "-o", "test_out_m0.lua", "-m", "0"]),
    ("Extreme memory limit one (-m 1)", ["-i", "Cargo.toml", "-o", "test_out_m1.lua", "-m", "1"]),
    ("Invalid memory limit string (-m abc)", ["-i", "Cargo.toml", "-o", "test_out.lua", "-m", "abc"]),
    ("Help flag (--help)", ["--help"]),
    ("Version flag (--version)", ["--version"]),
]

for name, args in tests:
    cmd = [bin_path] + args
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("=" * 60)
    print(f"TEST: {name}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"ExitCode: {res.returncode}")
    print(f"Stdout:\n{res.stdout.strip()}")
    print(f"Stderr:\n{res.stderr.strip()}")
