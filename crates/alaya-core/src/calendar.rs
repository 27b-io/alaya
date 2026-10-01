//! UTC civil dates from epoch seconds, without a date-time dependency.
//! Shared by the judge daily cap's log lines and the stats day buckets.

/// Returns (year, month, day) in UTC for a given Unix timestamp in seconds.
/// Implements Howard Hinnant's civil calendar algorithm (pure integer math).
pub fn utc_date(epoch_secs: u64) -> (i32, u32, u32) {
    let days = (epoch_secs / 86400) as i64;
    let z = days + 719468;
    let era = z / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m, d)
}

/// `YYYY-MM-DD` in UTC.
pub fn utc_date_str(epoch_secs: u64) -> String {
    let (y, m, d) = utc_date(epoch_secs);
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_date_str_computes_civil_calendar_correctly() {
        // Unix epoch start
        assert_eq!(utc_date_str(0), "1970-01-01");
        assert_eq!(utc_date_str(86399), "1970-01-01");
        assert_eq!(utc_date_str(86400), "1970-01-02");
        // Leap year 2024-02-29 (1709164800 is 2024-02-29 00:00:00 UTC)
        assert_eq!(utc_date_str(1709164800), "2024-02-29");
        assert_eq!(utc_date_str(1709251199), "2024-02-29");
        assert_eq!(utc_date_str(1709251200), "2024-03-01");
        // Known date: 2026-09-18
        assert_eq!(utc_date_str(1789733949), "2026-09-18");
    }
}
