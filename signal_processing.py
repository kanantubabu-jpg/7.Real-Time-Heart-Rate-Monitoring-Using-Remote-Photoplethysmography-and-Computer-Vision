import numpy as np
from scipy.signal import butter, filtfilt, detrend, welch, find_peaks

MIN_SIGNAL_SAMPLES = 45
HR_MIN_HZ = 0.75
HR_MAX_HZ = 3.5
BPM_MIN = 45
BPM_MAX = 120


def bandpass_filter(signal, fs, lowcut=HR_MIN_HZ, highcut=HR_MAX_HZ, order=3):
    signal = np.asarray(signal, dtype=np.float64)
    nyquist = 0.5 * fs
    low = max(lowcut / nyquist, 0.01)
    high = min(highcut / nyquist, 0.99)
    if low >= high or len(signal) < 12:
        return signal
    b, a = butter(order, [low, high], btype='band')
    padlen = min(len(signal) - 1, 3 * max(len(a), len(b)))
    if padlen < 1 or len(signal) <= padlen:
        return signal
    return filtfilt(b, a, signal, padlen=padlen)


def _normalize_signal(signal):
    signal = np.asarray(signal, dtype=np.float64)
    signal = detrend(signal, type='linear')
    std = np.std(signal)
    if std <= 1e-6:
        return None
    return (signal - np.mean(signal)) / std


def chrom_signal(rgb):
    """CHROM method - robust to motion and lighting changes."""
    rgb = np.asarray(rgb, dtype=np.float64)
    red = rgb[:, 0]
    green = rgb[:, 1]
    blue = rgb[:, 2]

    x_chrom = 3.0 * red - 2.0 * green
    y_chrom = 1.5 * red + green - 1.5 * blue
    std_x = np.std(x_chrom)
    std_y = np.std(y_chrom)
    alpha = std_x / std_y if std_y > 1e-6 else 0.0
    return x_chrom - alpha * y_chrom


def pos_signal(rgb):
    """POS method - plane orthogonal to skin tone."""
    rgb = np.asarray(rgb, dtype=np.float64)
    rgb_norm = rgb / (np.mean(rgb, axis=1, keepdims=True) + 1e-6)

    s1 = rgb_norm[:, 1] - rgb_norm[:, 2]
    s2 = rgb_norm[:, 1] + rgb_norm[:, 2] - 2.0 * rgb_norm[:, 0]
    std_s1 = np.std(s1)
    std_s2 = np.std(s2)
    alpha = std_s1 / std_s2 if std_s2 > 1e-6 else 0.0
    return s1 + alpha * s2


def _welch_spectrum(signal, fs):
    nperseg = min(len(signal), 256)
    nperseg = max(nperseg, 32)
    noverlap = max(0, nperseg // 2)
    freqs, power = welch(
        signal,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
    )
    return freqs, power


def _peak_score(freq, power_at_peak, freqs, power, prev_bpm=None):
    score = float(power_at_peak)

    half_freq = freq / 2.0
    if half_freq >= HR_MIN_HZ:
        half_idx = np.argmin(np.abs(freqs - half_freq))
        if power[half_idx] > power_at_peak * 0.55:
            score *= 0.45

    double_freq = freq * 2.0
    if double_freq <= HR_MAX_HZ:
        double_idx = np.argmin(np.abs(freqs - double_freq))
        if power[double_idx] > power_at_peak * 0.65:
            score *= 0.55

    if prev_bpm is not None:
        prev_hz = prev_bpm / 60.0
        freq_error = abs(freq - prev_hz)
        score *= np.exp(-((freq_error / 0.12) ** 2))

    return score


def _find_best_bpm(freqs, power, prev_bpm=None):
    valid = (freqs >= HR_MIN_HZ) & (freqs <= HR_MAX_HZ)
    if not np.any(valid):
        return None

    valid_freqs = freqs[valid]
    valid_power = power[valid]
    peak_indices, _ = find_peaks(
        valid_power,
        height=np.max(valid_power) * 0.25,
        distance=max(1, int(0.08 / (valid_freqs[1] - valid_freqs[0] + 1e-8))),
    )

    if len(peak_indices) == 0:
        peak_idx = int(np.argmax(valid_power))
        bpm = valid_freqs[peak_idx] * 60.0
        if BPM_MIN <= bpm <= BPM_MAX:
            return int(round(bpm))
        return None

    best_bpm = None
    best_score = -1.0
    for idx in peak_indices:
        freq = valid_freqs[idx]
        bpm = freq * 60.0
        if bpm < BPM_MIN or bpm > BPM_MAX:
            continue
        score = _peak_score(freq, valid_power[idx], valid_freqs, valid_power, prev_bpm)
        if score > best_score:
            best_score = score
            best_bpm = int(round(bpm))

    return best_bpm


def estimate_bpm_from_signal(signal, fs, prev_bpm=None):
    if len(signal) < MIN_SIGNAL_SAMPLES or fs <= 0:
        return None

    normalized = _normalize_signal(signal)
    if normalized is None:
        return None

    try:
        filtered = bandpass_filter(normalized, fs)
    except (ValueError, np.linalg.LinAlgError):
        filtered = normalized

    freqs, power = _welch_spectrum(filtered, fs)
    return _find_best_bpm(freqs, power, prev_bpm)


def estimate_bpm_from_rgb(rgb, fs, prev_bpm=None):
    if len(rgb) < MIN_SIGNAL_SAMPLES or fs <= 0:
        return None

    chrom = _normalize_signal(chrom_signal(rgb))
    pos = _normalize_signal(pos_signal(rgb))
    if chrom is None and pos is None:
        return None

    candidates = []
    for signal in (chrom, pos):
        if signal is None:
            continue
        try:
            filtered = bandpass_filter(signal, fs)
        except (ValueError, np.linalg.LinAlgError):
            filtered = signal
        freqs, power = _welch_spectrum(filtered, fs)
        bpm = _find_best_bpm(freqs, power, prev_bpm)
        if bpm is not None:
            candidates.append(bpm)

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    if abs(candidates[0] - candidates[1]) <= 8:
        return int(round(np.mean(candidates)))

    return candidates[0]


def estimate_bpm(signal, fs, prev_bpm=None):
    return estimate_bpm_from_signal(signal, fs, prev_bpm)


def signal_quality_score(signal, fs):
    if len(signal) < MIN_SIGNAL_SAMPLES or fs <= 0:
        return 0.0

    normalized = _normalize_signal(signal)
    if normalized is None:
        return 0.0

    try:
        filtered = bandpass_filter(normalized, fs)
    except (ValueError, np.linalg.LinAlgError):
        filtered = normalized

    freqs, power = _welch_spectrum(filtered, fs)
    total_power = np.sum(power) + 1e-8
    band_mask = (freqs >= HR_MIN_HZ) & (freqs <= HR_MAX_HZ)
    band_power = np.sum(power[band_mask])
    peak_power = np.max(power[band_mask]) if np.any(band_mask) else 0.0

    ratio = band_power / total_power
    peak_ratio = peak_power / (total_power + 1e-8)
    score = 0.65 * min(1.0, ratio * 1.5) + 0.35 * min(1.0, peak_ratio * 8.0)
    return min(1.0, max(0.0, score))


def signal_quality_score_rgb(rgb, fs):
    if len(rgb) < MIN_SIGNAL_SAMPLES or fs <= 0:
        return 0.0

    chrom = _normalize_signal(chrom_signal(rgb))
    pos = _normalize_signal(pos_signal(rgb))
    scores = []
    for signal in (chrom, pos):
        if signal is not None:
            scores.append(signal_quality_score(signal, fs))

    if not scores:
        return 0.0
    return float(np.mean(scores))


def reject_bpm_outlier(new_bpm, history, max_jump=10):
    if new_bpm is None or len(history) < 3:
        return new_bpm
    median_bpm = np.median(history)
    if abs(new_bpm - median_bpm) > max_jump:
        return None
    return new_bpm
