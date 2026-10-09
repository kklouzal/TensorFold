#pragma once

// Tile identity, geometry and compile-time modes have one source for dispatch
// and the host's exact packed-read projection. No floating-point policy lives here.
#define TENSORFOLD_QMM_GROUP_TILES(X) \
    X(1,  16,  64, 1, 4, 4, false, false) \
    X(2,  16,  64, 1, 4, 8, false, false) \
    X(3,  32,  64, 1, 4, 4, false, false) \
    X(4,  64,  64, 1, 4, 4, false, false) \
    X(5,  64, 128, 2, 4, 3, false, false) \
    X(6,   8,  64, 1, 4, 4, true,  false) \
    X(7,   8, 128, 1, 4, 4, true,  false) \
    X(8,  16, 128, 1, 8, 4, false, false) \
    X(9,  32, 128, 1, 8, 4, false, false) \
    X(10,128, 128, 2, 4, 2, false, true) \
    X(11, 64, 128, 1, 8, 3, false, true) \
    X(12, 64, 128, 2, 4, 3, false, true)

#define TENSORFOLD_QMM_PREFILL_TILES(X) \
    X(0, 128, 128, 2, 4, 3) \
    X(1,  64, 128, 1, 4, 4) \
    X(2, 128,  64, 2, 2, 4) \
    X(3,  64,  64, 1, 4, 4) \
    X(4, 128, 128, 2, 4, 4) \
    X(5, 128, 128, 2, 2, 3) \
    X(6, 128, 128, 2, 2, 4) \
    X(7, 128, 256, 2, 4, 3) \
    X(8,  64, 256, 1, 4, 4) \
    X(9, 128, 128, 2, 2, 2) \
    X(10, 64, 128, 1, 2, 2) \
    X(11,128, 256, 2, 4, 2)

namespace tensorfold {

struct QMMTile {
    int rows;
    int columns;
    bool pairs;
};

inline int resolve_qmm_group_tile(int tile, int rows, bool gb10) {
    if (tile != 0) return tile;
    if (gb10) return rows <= 16 ? 2 : rows <= 32 ? 3 : rows <= 64 ? 4 : 5;
    return rows <= 8 ? 7 : rows <= 16 ? 8 : rows <= 32 ? 9 : rows <= 64 ? 4 : 5;
}

inline QMMTile qmm_group_tile(int tile) {
    switch (tile) {
#define GROUP_COLUMNS(ID, BM, BN, WM, WN, STAGES, SWAP, SPREAD) case ID: return {BM, BN, !SWAP};
        TENSORFOLD_QMM_GROUP_TILES(GROUP_COLUMNS)
#undef GROUP_COLUMNS
        default: return {0, 0, true};
    }
}

inline QMMTile qmm_prefill_tile(int tile) {
    switch (tile) {
#define PREFILL_COLUMNS(ID, BM, BN, WM, WN, STAGES) case ID: return {BM, BN, true};
        TENSORFOLD_QMM_PREFILL_TILES(PREFILL_COLUMNS)
#undef PREFILL_COLUMNS
        default: return {128, 128, true};
    }
}

} // namespace tensorfold
