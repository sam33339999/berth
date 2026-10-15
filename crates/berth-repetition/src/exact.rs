//! Exact consecutive repetition over one channel window.
//!
//! A qualifying run is `count >= 2` adjacent copies of one block of `len` tokens,
//! ending at the newest accepted token. A single different token between copies
//! means they are not consecutive, so the pair does not qualify.
//!
//! When several `(len, count)` pairs qualify, the pair reported here is the
//! smallest `len` that is at least the configured minimum, with the largest
//! `count` of that block. Larger multiples of the same run are not also
//! reported at that step. The caller keeps one label per `(len, channel)` and
//! rewrites that label when `count` grows.
//!
//! Cost: one Z-array over the reversed window. That is O(W) time per accepted
//! token and O(W) scratch, with W <= [`crate::MAX_WINDOW`] (4096). Appending a
//! token changes the reversed prefix, so the Z-array is rebuilt rather than
//! patched; the cost stays bounded by the window, not by the generation length.
//! A minimum above W/2 never matches: two copies have to fit in the window.
//! The exact-repeat window is this cap, not the tail-check lookback. The tail
//! lookback only trims the slice passed to [`crate::TailDetector`].

/// Smallest qualifying period at the end of `window`, if any.
///
/// `reversed` and `z` are scratch and are cleared here. `min_len == 0` never matches.
pub(crate) fn best_run(
    window: &[u32],
    min_len: u32,
    reversed: &mut Vec<u32>,
    z: &mut Vec<usize>,
) -> Option<(u32, u32)> {
    let n = window.len();
    let min_len = min_len as usize;
    if min_len == 0 || n / 2 < min_len {
        return None;
    }
    reversed.clear();
    reversed.extend(window.iter().rev().copied());
    z_array(reversed, z);
    let half = n / 2;
    for (period, &matched) in z.iter().enumerate().take(half + 1).skip(min_len) {
        if matched >= period {
            // Z[p] >= p means the reversed prefix is at least two copies of the
            // p-block. Each extra full copy adds p to the match length.
            let count = 1 + matched / period;
            let Ok(period) = u32::try_from(period) else {
                return None;
            };
            let Ok(count) = u32::try_from(count) else {
                return None;
            };
            return Some((period, count));
        }
    }
    None
}

/// Linear-time Z-array. `z[0]` is left 0. `z[i]` is the longest prefix of `text`
/// that also occurs at `i`.
fn z_array(text: &[u32], z: &mut Vec<usize>) {
    let n = text.len();
    z.clear();
    z.resize(n, 0);
    let mut left = 0usize;
    let mut right = 0usize;
    for i in 1..n {
        if i < right {
            z[i] = (right - i).min(z[i - left]);
        }
        while i + z[i] < n && text[z[i]] == text[i + z[i]] {
            z[i] += 1;
        }
        if i + z[i] > right {
            left = i;
            right = i + z[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear(window: &[u32], min_len: u32) -> Option<(u32, u32)> {
        best_run(window, min_len, &mut Vec::new(), &mut Vec::new())
    }

    /// Independent of the Z-array. Smallest `p >= min_len` whose block repeats
    /// at the end, then the longest such run.
    fn naive(window: &[u32], min_len: u32) -> Option<(u32, u32)> {
        let n = window.len();
        let min_len = min_len as usize;
        if min_len == 0 || n / 2 < min_len {
            return None;
        }
        for period in min_len..=n / 2 {
            let mut k = 1usize;
            while (k + 1) * period <= n
                && window[n - (k + 1) * period..n - k * period] == window[n - period..n]
            {
                k += 1;
            }
            if k >= 2 {
                return Some((period as u32, k as u32));
            }
        }
        None
    }

    #[test]
    fn matches_block_compare_on_scripted_windows() {
        let samples: &[&[u32]] = &[
            &[],
            &[1],
            &[1, 1],
            &[1, 1, 1, 1, 1, 1],
            &[1, 2, 1, 2, 1, 2],
            &[1, 2, 3, 1, 2, 3, 1, 2, 3],
            &[1, 2, 3, 9, 1, 2, 3],
            &[1, 2, 3, 1, 2, 3, 1, 2],
            &[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            &[7, 7, 7, 8, 8, 8],
        ];
        for window in samples {
            for min_len in 0..6 {
                assert_eq!(
                    linear(window, min_len),
                    naive(window, min_len),
                    "window={window:?} min_len={min_len}"
                );
            }
        }
    }

    #[test]
    fn matches_block_compare_on_a_deterministic_stream() {
        let mut state = 0x1234_5678u32;
        let mut window = Vec::new();
        for _ in 0..40 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let op = state % 5;
            if op == 0 && !window.is_empty() {
                window.pop();
            } else {
                window.push(state % 4);
            }
            if window.len() > 48 {
                window.remove(0);
            }
            for min_len in 1..5 {
                assert_eq!(linear(&window, min_len), naive(&window, min_len));
            }
        }
    }

    #[test]
    fn sixty_four_block_three_times_reports_64x3_not_a_larger_period() {
        let block: Vec<u32> = (0..64).collect();
        let mut window = Vec::new();
        window.extend(block.iter().copied());
        window.extend(block.iter().copied());
        assert_eq!(linear(&window, 64), Some((64, 2)));
        window.extend(block.iter().copied());
        assert_eq!(linear(&window, 64), Some((64, 3)));
        // 128 tokens do not fit twice inside 192.
        assert_ne!(linear(&window, 64), Some((128, 2)));
    }

    #[test]
    fn minimum_above_the_primitive_period_uses_the_next_multiple() {
        let mut window = Vec::new();
        for _ in 0..6 {
            window.extend([1, 2, 3]);
        }
        assert_eq!(linear(&window, 1), Some((3, 6)));
        assert_eq!(linear(&window, 6), Some((6, 3)));
    }
}
