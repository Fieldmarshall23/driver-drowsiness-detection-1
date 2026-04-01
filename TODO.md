# Fixing RHDA Error: Install Dependencies in Virtual Environment

## Steps:
- [x] 1. ✓ 'drowzy/' is valid venv (pyvenv.cfg, Scripts/).

# Skipped 2 (already exists)

- [x] 3. ✓ Activated venv `(drowzy)`

- [x] 4. ✓ All deps installed ("Requirement already satisfied" for uvicorn[standard], fastapi, tensorflow==2.12.0, dlib==20.0.0, etc.)

- [ ] 5. Verify: `python test_imports.py` (running...)

- [ ] 6. VSCode: 
  - Reload window (Ctrl+Shift+P > Developer: Reload Window).
  - Python: Select Interpreter > `c:/Users/kmaka/Desktop/project/driver-drowsiness-detection/drowzy/Scripts/python.exe`.
  - Optional: Add to `.vscode/settings.json`: `{ "python.defaultInterpreterPath": ".\\drowzy\\Scripts\\python.exe", "python.analysis.useVirtualEnv": true }`

- [ ] 7. Deactivate: `deactivate` (when done).

