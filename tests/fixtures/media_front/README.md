`colors.mkv` is an independently authored lossless five-frame RGB fixture, 64x48
at 2 FPS. Colors in order: (255,0,0), (0,255,0), (0,0,255), (23,47,89),
(255,255,255). Each frame is uniform. SHA256:
`b3c4fce96a90b35f6b7e7b1afd9e98dd138eff7f837127eece25db5233a313e5`.

Encoded once with host FFmpeg 8.0.1-3ubuntu2, FFV1 bgr0 in Matroska, from the
specified raw RGB bytes. Tests require only the pinned CPU decoder, without an
encoder or host FFmpeg dependency. Expected RGB frames and timestamps are
specified independently of PyAV and the selected decoder.

`colors-vfr.mkv` contains the same five frames with a one-second gap inserted
before frame 2 (`setpts=PTS+if(gte(N,2),1,0)/TB`, `-fps_mode vfr`). Actual PTS in
seconds: 0, 0.5, 2, 2.5, 3. The declared nominal FPS remains 2. SHA256:
`6fb79ff9da1c05eaaf94b587e8f1b66c2d06121c69493da4e659cd3c8bbcf68b`.
