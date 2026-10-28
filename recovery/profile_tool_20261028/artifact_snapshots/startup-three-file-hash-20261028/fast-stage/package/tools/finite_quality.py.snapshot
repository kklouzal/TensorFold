"""Small deterministic answer/format spot checks, not broad quality qualification.

Every answer is independently specified. Only outer whitespace is ignored;
prompts explicitly require the exact answer format, including minified JSON.
No model output is executed or used as a reference answer.
"""
import sys

VERSION = "finite-quality-v1"
OUTPUT_TOKENS = 320
CASES = (
    ("math", "A workshop has 17 boxes containing 24 bolts each. It ships 135 bolts, then receives "
     "48 bolts. How many bolts remain? Return only the integer, without explanation.", "321"),
    ("retrieval", "Use only this synthetic access table:\n"
     "Cedar RED: P4N7\nBirch GREEN: T8K2\nAster RED: W3J6\nCedar GREEN: R8Q2\n"
     "Dune GREEN: M9L1\nBirch RED: H5V4\n"
     "What is Cedar's GREEN access code? Return only the code, without explanation.", "R8Q2"),
    ("logic", "Exactly one of A, B, and C is true. If A is true, B is true. C is false. "
     "Which letter is true? Return only that uppercase letter, without explanation.", "B"),
    ("instruction", "Given the ordered list cobalt, amber, jade, silver, return items 2 and 4 "
     "using 1-based positions. Convert them to uppercase and join them with one slash. "
     "Return only the resulting string.", "AMBER/SILVER"),
    ("json", "A shop starts with 9 rolls of tape, receives 4, and sells 6. Its labels are green "
     "and blue. Return one minified JSON object with keys item, remaining, labels in exactly "
     "that order. Set item to the string tape, remaining to the remaining integer count, and "
     "labels to the two label strings sorted alphabetically. Use no extra text or whitespace "
     "inside the JSON.", '{"item":"tape","remaining":7,"labels":["blue","green"]}'),
    ("code-output", "What does this Python 3 program print? Return only its single output line, "
     "without code fences or explanation.\n"
     "a = [2, 5, 1]\nb = a[:]\na[1] = 7\nb.append(a.pop(0))\nprint(sum(a), sum(b), len(a))",
     "8 10 2"),
)


def control_payload_bytes():
    """Charge the fixed tuple/string oracle payload using the selected runtime.

    Prepared IDs/source buffers, reply text and bounded report graphs use the
    existing finite caller terms. This is additional static oracle/control data.
    """
    return sys.getsizeof(CASES) + sum(sys.getsizeof(case) + sum(sys.getsizeof(part) for part in case)
                                    for case in CASES)


def assess(case_id, text):
    expected = next((answer for name, _, answer in CASES if name == case_id), None)
    if expected is None:
        raise ValueError("unknown fixed quality case")
    if type(text) is not str:
        raise TypeError("quality assessment requires the bounded decoded response")
    observed = text.strip()
    passed = observed == expected
    difference = None
    if not passed:
        offset = next((index for index, pair in enumerate(zip(expected, observed)) if pair[0] != pair[1]),
                      min(len(expected), len(observed)))
        difference = {"offset": offset, "expected_slice": expected[offset:offset + 80],
                      "observed_slice": observed[offset:offset + 80], "observed_characters": len(observed)}
    return {"passed": passed, "assessment_complete": True, "expected": expected, "difference": difference,
            "comparison": "required exact answer format; outer whitespace ignored"}
