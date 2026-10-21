"""Original public calibration errors versus explicit invalid preflight states."""

import ast
import math
import struct
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "src/tensorfold/cuda/nvfp4/format.py"
nodes = [node for node in ast.parse(SOURCE.read_bytes()).body
         if isinstance(node, ast.FunctionDef) and node.name in {"_calibration_result", "calibration"}]
scope = {}
exec(compile(ast.Module(nodes, []), str(SOURCE), "exec"), scope)  # noqa: S102 - actual owned scalar boundary, no external code.


@pytest.mark.parametrize("reciprocal", [False, True])
@pytest.mark.parametrize("value", [-1.0, -0.0, 0.0, float("inf"), float("nan"), 2.0**-150,
                                  2.0**-149, 2.0**-127, 2.0**-126, 1.0, 3.0, 2.0**128])
def test_public_exact_value_or_original_valueerror_is_preserved_without_exception_eligibility(value, reciprocal):
    if not math.isfinite(value) or value <= 0:
        error, expected = "calibration must be finite and positive", None
    else:
        expected = 1.0 / value if reciprocal else value
        try:
            narrowed = struct.unpack("f", struct.pack("f", expected))[0]
        except OverflowError:
            narrowed = float("inf")
        error = None if math.isfinite(narrowed) and narrowed > 0 else "calibration must remain finite and positive in FP32"
    result, reason = scope["_calibration_result"](value, reciprocal=reciprocal)
    assert reason == error
    if error is not None:
        assert result is None
        with pytest.raises(ValueError) as caught:
            scope["calibration"](value, name="projection.field", reciprocal=reciprocal)
        assert type(caught.value) is ValueError and str(caught.value) == "projection.field: " + error
    else:
        assert struct.pack("d", result) == struct.pack("d", expected)
        assert scope["calibration"](value, name="projection.field", reciprocal=reciprocal) == expected


def test_unexpected_conversion_failure_remains_the_original_exception():
    primary = ValueError("conversion defect")

    class Scalar:
        def __float__(self):
            raise primary

    for function, kwargs in ((scope["_calibration_result"], {}), (scope["calibration"], {"name": "field"})):
        with pytest.raises(ValueError) as caught:
            function(Scalar(), **kwargs)
        assert caught.value is primary
