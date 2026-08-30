//! Numbers spelled the way CPython spells them.
//!
//! Only ever for text a human reads -- error messages, diagnostics, the
//! renderers. The reference's `ValueError` strings are part of its contract
//! (`tests/test_graph_conditioning.py` matches on one), and Rust's formatter
//! disagrees with Python's in two places that show up immediately: `{:e}`
//! writes `1e12` where Python writes `1e+12`, and `{}` writes `1` where
//! Python writes `1.0`.

/// `"%.*e" % x`: a signed exponent of at least two digits.
pub fn sci(x: f64, places: usize) -> String {
    pad_exponent(&format!("{:.*e}", places, x))
}

/// `repr(x)` for a float: the shortest string that reads back, with a `.0` on
/// anything that would otherwise look like an int, and an exponent outside
/// `[1e-4, 1e16)`.
pub fn float(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let magnitude = x.abs();
    if x != 0.0 && (magnitude < 1e-4 || magnitude >= 1e16) {
        // No `.0` on the mantissa here: CPython writes `1e-05`, not `1.0e-05`.
        return pad_exponent(&format!("{x:e}"));
    }
    let plain = format!("{x}");
    if plain.contains('.') {
        plain
    } else {
        format!("{plain}.0")
    }
}

/// `f"{x:g}"`: six significant digits, trailing zeros stripped, and no
/// decimal point on a whole number. `PIN_LADDER` prints through this, so a
/// candidate's label is `x0.125` and `x1` rather than `x0.125` and `x1.0`.
pub fn general(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if x == 0.0 {
        return "0".to_string();
    }
    let exponent = x.abs().log10().floor() as i32;
    // CPython's `%g` switches to exponent form outside [-4, precision).
    if exponent < -4 || exponent >= 6 {
        let raw = format!("{:.*e}", 5, x);
        let (mantissa, rest) = raw.split_once('e').unwrap();
        return pad_exponent(&format!("{}e{}", strip(mantissa), rest));
    }
    strip(&format!("{:.*}", (5 - exponent).max(0) as usize, x))
}

/// Trailing zeros after a decimal point, and the point itself if nothing is
/// left after it.
fn strip(text: &str) -> String {
    if !text.contains('.') {
        return text.to_string();
    }
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// `f"{x:.Nf}"`. Rust agrees with Python on this one -- both round ties to
/// even -- but `inf` prints as `inf` on both sides only because this says so.
pub fn fixed(x: f64, places: usize) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    format!("{:.*}", places, x)
}

fn pad_exponent(raw: &str) -> String {
    match raw.split_once('e') {
        Some((mantissa, exponent)) => {
            let (sign, digits) = match exponent.strip_prefix('-') {
                Some(rest) => ('-', rest),
                None => ('+', exponent),
            };
            format!("{mantissa}e{sign}{digits:0>2}")
        }
        None => raw.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponents_carry_a_sign_and_two_digits() {
        assert_eq!(sci(1e12, 0), "1e+12");
        assert_eq!(sci(1.2345e15, 3), "1.234e+15");
        assert_eq!(sci(5e-3, 3), "5.000e-03");
    }

    #[test]
    fn general_strips_what_g_strips() {
        assert_eq!(general(0.0), "0");
        assert_eq!(general(1.0), "1");
        assert_eq!(general(0.125), "0.125");
        assert_eq!(general(0.5), "0.5");
        assert_eq!(general(4.0), "4");
        assert_eq!(general(1.0 / 3.0), "0.333333");
        assert_eq!(general(1e-5), "1e-05");
        assert_eq!(general(1234567.0), "1.23457e+06");
    }

    #[test]
    fn whole_floats_keep_their_point() {
        assert_eq!(float(0.0), "0.0");
        assert_eq!(float(-1.0), "-1.0");
        assert_eq!(float(1.5), "1.5");
        assert_eq!(float(1e-5), "1e-05");
        assert_eq!(float(1e16), "1e+16");
        assert_eq!(float(1e15), "1000000000000000.0");
        assert_eq!(float(f64::NAN), "nan");
        assert_eq!(float(f64::NEG_INFINITY), "-inf");
    }
}
