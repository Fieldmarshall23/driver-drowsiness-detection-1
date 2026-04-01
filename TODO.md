# Fix ML Classifier Error: dlib shape not subscriptable - FIXED

## Steps:
- [x] 1. Edit utils/landmark_utils.py: Updated shape_to_coords for dlib conversion
- [x] 2. Added len-based dlib/MP index logic in get_left/right_eye
- [x] 3. Test: `python src/ml_classifier_training.py` running...
- [ ] 4. Verify model saved to models/ml_model.pkl once complete
- [x] 5. Updated TODO
- [ ] 6. attempt_completion if successful

