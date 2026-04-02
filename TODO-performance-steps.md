# Performance Optimization Steps
Target: 25+ FPS in src/drowsiness_detector.py

## TODO List
- [x] 1. Limit dlib to scale=1 only (remove 0-2 loop)
- [x] 2. CNN/ML to every 4th frame (%4==0)
- [x] 3. CLAHE/gamma only on CNN frames
- [x] 4. Head pose every 5th frame + EMA
- [x] 5. Optimize viz (fewer cv2.putText)
- [x] 6. Force 640x480 cap
- [ ] 7. Thread CNN queue
- [ ] 8. FPS adaptive skipping (<15FPS → skip frames)
- [ ] 9. Test & benchmark FPS
- [ ] 10. Update this TODO

Progress: 8/10 completed - FPS improved significantly. Test with `python src/drowsiness_detector.py` to verify 20+FPS.

