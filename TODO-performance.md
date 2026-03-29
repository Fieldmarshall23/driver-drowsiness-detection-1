# Performance Optimization TODO

**Goal**: Real-time face detection without sluggishness (target FPS 30+).

**Current FPS**: 0.3-2.6 (logs show).

**Plan Steps**:
- [ ] Skip CNN/ML every frame (use every 3rd or prob-based).
- [ ] Limit dlib to scale 0 (fastest).
- [ ] Haar primary for faces (faster).
- [ ] Remove CLAHE/gamma if slowing.
- [ ] Test & measure FPS.

Proceed?
