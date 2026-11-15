"""The grouped 4-bit lane matmul's block by rows and chip (host only; blocks never change bits)."""

WIDE_SMS = 96               # SM 12.0 GPUs from this many SMs (the RTX PRO 6000's 188) take the wide blocks


def group_tile(m: int, major: int, minor: int, sms: int) -> int:
    """0 (the kernel's own pick by rows and chip) or a wide block for ``m`` rows on SM major.minor with ``sms`` SMs."""

    if (major, minor) != (12, 0) or sms < WIDE_SMS or m <= 16:
        return 0
    return 12 if m <= 32 else 11 if m <= 64 else 10 if m <= 96 else 11


def capped_row_tile(m: int) -> int:
    """Rows a program takes: 16 to 128, then tiles of 128 (a row's bits never depend on its tile)."""

    for b in (16, 32, 64, 128):
        if m <= b:
            return b
    return 128
