import numpy as np
import sounddevice as sd
import sounddevice as sd

# ------------ CONFIG ------------
SAMPLE_RATE = 44100      # Hz
DURATION = 2.5           # seconds to record
MIN_F0 = 60.0            # minimum expected pitch (Hz)
MAX_F0 = 400.0           # maximum expected pitch (Hz)
THRESHOLD_HZ = 165.0     # below = male, above = female (adjust as needed)
# -------------------------------


def record_audio(duration=DURATION, samplerate=SAMPLE_RATE):
    print(f"Recording for {duration:.1f} s at {samplerate} Hz...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    print("Recording finished.")
    return audio

import librosa

def estimate_pitch_pyin(signal, samplerate=SAMPLE_RATE, fmin=MIN_F0, fmax=MAX_F0):
    """
    Pitch estimator using librosa's pYIN algorithm — much more robust
    than raw autocorrelation, especially with background noise.
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        signal.astype(np.float32),
        fmin=fmin,
        fmax=fmax,
        sr=samplerate
    )

    # Keep only frames librosa is confident are voiced speech
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) == 0:
        return None

    return float(np.median(voiced_f0))

def estimate_pitch_autocorr(signal, samplerate=SAMPLE_RATE,
                            fmin=MIN_F0, fmax=MAX_F0):
    """
    Pitch estimator using autocorrelation, with basic noise rejection.
    """
    sig = signal - np.mean(signal)

    # Check signal has enough energy (not silence/noise floor)
    rms = np.sqrt(np.mean(sig**2))
    if rms < 0.01:
        return None

    corr = np.correlate(sig, sig, mode='full')
    corr = corr[len(corr) // 2:]

    min_lag = int(samplerate / fmax)
    max_lag = int(samplerate / fmin)
    if max_lag <= min_lag + 1:
        return None

    corr_segment = corr[min_lag:max_lag]

    peak_index = int(np.argmax(corr_segment))
    peak_value = corr_segment[peak_index]

    # Reject weak/noisy peaks
    if corr[0] <= 0 or peak_value < 0.3 * corr[0]:
        return None

    # Reject peaks sitting right at the search boundary (likely false lock)
    if peak_index <= 1 or peak_index >= len(corr_segment) - 2:
        return None

    best_lag = peak_index + min_lag
    f0 = samplerate / best_lag
    return float(f0)


def classify_gender(f0, threshold_hz=THRESHOLD_HZ):
    if f0 is None:
        return None
    return "female" if f0 >= threshold_hz else "male"


def main():
    try:
        audio = record_audio()
    except Exception as e:
        print("Error recording audio:", e)
        return

    print("Estimating pitch, please wait...")
    f0 = estimate_pitch_autocorr(audio)

    if f0 is None or np.isnan(f0):
        print("Could not detect a clear pitch.")
        print("Tips: Speak a steady 'aaaa' sound, closer to the mic, in a quiet room.")
        return

    gender = classify_gender(f0)

    print("\n-----------------------------")
    print(f"Estimated pitch (F0): {f0:.1f} Hz")
    print(f"Threshold used:       {THRESHOLD_HZ:.1f} Hz")
    if gender is None:
        print("Classification: Not sure.")
    else:
        print(f"Classification:       {gender.upper()}")
    print("-----------------------------\n")


if __name__ == "__main__":
    main()
