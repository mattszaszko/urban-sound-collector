"""Audio sample-rate and chunk window constants."""

# Native INMP441 / I2S capture rate (locked).
CAPTURE_SAMPLE_RATE = 48_000

# Bundled YAMNet TFLite model expects 15,600 mono samples @ 16 kHz (0.975 s).
YAMNET_SAMPLE_RATE = 16_000
YAMNET_CHUNK_DURATION_SECONDS = 0.975
YAMNET_CHUNK_SAMPLES = 15_600

# One collector chunk at the native capture rate (~0.975 s @ 48 kHz).
CAPTURE_CHUNK_SAMPLES = int(
    round(CAPTURE_SAMPLE_RATE * YAMNET_CHUNK_DURATION_SECONDS)
)

# Backwards-compatible aliases.
TARGET_SAMPLE_RATE = YAMNET_SAMPLE_RATE
CHUNK_DURATION_SECONDS = YAMNET_CHUNK_DURATION_SECONDS
CHUNK_SAMPLES = YAMNET_CHUNK_SAMPLES
