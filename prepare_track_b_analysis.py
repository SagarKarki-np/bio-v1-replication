
from pathlib import Path

analysis = Path("analysis")

files = [
    "figure5_interpolation_diagnostic.py",
    "figure5_interpolation_refit.py",
]

replacements = {
    '"track_a"': '"track_b"',
    '"track_a_100000"': '"track_b_100000"',
    '"interpolation_sensitivity_track_a"':
        '"interpolation_sensitivity_track_b"',
}

for filename in files:

    source = analysis / filename

    target = analysis / filename.replace(
        ".py", "_track_b.py"
    )

    code = source.read_text(encoding="utf-8")

    for old, new in replacements.items():

        if old not in code:
            raise ValueError(
                f"Expected {old} not found in {source}"
            )

        code = code.replace(old, new)

    target.write_text(code, encoding="utf-8")

    print("Created:", target)
