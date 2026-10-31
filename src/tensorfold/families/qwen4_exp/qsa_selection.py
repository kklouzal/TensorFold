"""Selected finite-FP32 target/MTP QSA order, independent of physical layout.

Both SM70 implementations select descending scores with ascending logical
pooled-group IDs at equal scores. Survivors reach attention in logical order
to preserve reduction ordering. This identity changes prefix and measured
execution compatibility; it grants no nonfinite/range or device qualification.
"""

POLICY = "qsa-fp32-score-desc-logical-group-asc-v1"
