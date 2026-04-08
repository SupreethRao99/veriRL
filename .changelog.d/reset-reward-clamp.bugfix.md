Fixed reset() returning reward=0.0, which violated the strict (0, 1) score range required by the validator; reset now returns reward=0.01.
