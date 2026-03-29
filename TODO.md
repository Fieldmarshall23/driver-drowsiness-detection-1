# TODO: Improve Drowsiness Detection Logic (src/drowsiness_detector.py)
Track progress: Updated as steps complete

## Step 1: [x] Complete - Add New Constants & Globals
- Added PERCLOS_WINDOW=300, EMA_ALPHA=0.3, FUSION_WEIGHTS=[0.3,0.25,0.2,0.15,0.1]
- EAR_BASELINE_FRAMES=100
- Globals: perclos_counter, ear_baseline=0.27, baseline_collected=False, ema_drowsy_prob=0.0, ema_eye_prob=0.5, ema_yawn_prob=0.0

## Step 2: [x] Complete - Baseline EAR Collection (first 100 frames)
- In detect_drowsiness(): if not baseline_collected and frame_counter < 100: accumulate EARs, set ear_baseline=np.mean

## Step 3: [x] Complete - Compute MAR & Update EMAs
- MAR from mouth landmarks (similar to EAR formula)
- ema_* = EMA_ALPHA * new + (1-EMA_ALPHA) * old

## Step 4: [x] Complete - PERCLOS & Adaptive Thresholds
- if eye_closed: perclos_counter +=1 else: perclos_counter=max(0,perclos_counter-1)
- perclos = perclos_counter / PERCLOS_WINDOW
- EAR_THRESHOLD = ear_baseline * 0.8 if baseline else 0.23

## Step 5: [x] Complete - Improved Fusion
- Normalize factors
- drowsy_prob = sum(weights * factors)

## Step 6: [ ] Smoother Counters & Alerts
- Decay: ear_counter *=1.02 if bad else *=0.95, cap at 50
- Raise consec: EAR_CONSEC_FRAMES=30, EYE_CNN_CONSEC=10
- Hysteresis: DROWSY_PROB_THRESH=0.65, RESET=0.35

## Step 7: [x] Complete - Update Prints/Logs
- Added EMA/PERCLOS to debug prints and overlays

## Step 8: [ ] Test & Tune
- Run detector, simulate cases, adjust via logs

