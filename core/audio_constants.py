"""YAMNet audio window constants for live capture."""

# YAMNet expects 16 kHz mono float32 PCM.
TARGET_SAMPLE_RATE = 16_000
# Official YAMNet window: 0.975 seconds = 15,600 samples at 16 kHz.
CHUNK_DURATION_SECONDS = 0.975
CHUNK_SAMPLES = 15_600
