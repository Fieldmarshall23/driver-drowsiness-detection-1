# Debug Drowsy Alert Not Detecting - Progress Tracker

## Approved Plan Steps:
1. [x] Create TODO.md with steps ✓
2. [x] Rename models/eye_cnn_model.h5 to models/cnn_model.h5 (fix load path mismatch) ✓ Confirmed exists
3. [x] Edit src/drowsiness_detector.py: 
   - Add try/except around model loads with logging/print ✓
   - Temporarily lower thresholds: DROWSY_PROB_THRESH=0.50, EAR_CONSEC_FRAMES=12, EYE_CNN_CONSEC=5, YAWN_CONSEC=2 ✓
   - Increase debug prints (every 10 frames instead of 30) ✓
   - Add print on alert trigger/counters ✓
4. [ ] Test run: python src/drowsiness_detector.py → simulate (close eyes 5s+, yawn), observe console probs/counters/alerts, check alerts.log
5. [ ] Analyze: low probs? → retrain; no faces? → camera; etc.
6. [ ] Revert temp changes
7. [ ] Complete

**Debug fixes complete (lower thresh 0.50, every 10 frame prints, model error handling). TODO steps 1-3 ✓. Rename cnn_model.h5 ✓. Test running.**

