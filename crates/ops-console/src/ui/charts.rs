//! Charts: the console's key values drawn as server-rendered SVG.
//!
//! The CSP admits no script and no inline style, so a chart is plain SVG
//! markup: its geometry is presentation attributes and its colour a theme
//! token class (`fill-chart-1`), which is also what makes it follow dark
//! mode. No `viewBox` scales a chart's marks. Positions along the width are
//! percentages of the chart's own width and heights are pixels, so a chart
//! fills any container without stretching its text, its rounded ends or the
//! gaps between its marks.
//!
//! Every chart is one `role="img"` element whose `aria-label` states its
//! numbers, each mark carries a native `<title>` tooltip, and the numbers
//! are printed beside the marks too, in a readout, a legend or axis ticks.
//! A value that is missing (`None`), not finite or negative is "no data",
//! drawn as such and never as a zero; a zero is drawn as a zero.
//!
//! No page draws a chart yet; the pages adopt them in their own changes.
//! Until then only the tests and the snapshot gallery use this module, and
//! the expectation below fails the lint once every item is in use, so it
//! cannot outlive its reason.
#![cfg_attr(not(test), expect(dead_code, reason = "no page draws a chart yet"))]

use leptos::either::Either;
use leptos::prelude::*;

// ─── Colour ─────────────────────────────────────────────────────────────────

/// A mark's colour, named by the job it does: a series' identity, a step on
/// an ordered scale, or what is not a series of its own. The values are the
/// `--chart-*` tokens in `style/input.css`, which records how they were
/// chosen and checked in both themes. Each holds its whole class name, so
/// Tailwind finds it in the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Colour(&'static str);

impl Colour {
    /// A series' identity. Give series their slots in display order from the
    /// first, never skipping one and never by rank, and keep an entity on its
    /// slot in every chart: the order is what keeps neighbours apart for
    /// colour-blind readers. Past eight series, fold the rest into one part
    /// in `OTHER`.
    pub const SERIES: [Colour; 8] = [
        Colour("fill-chart-1"),
        Colour("fill-chart-2"),
        Colour("fill-chart-3"),
        Colour("fill-chart-4"),
        Colour("fill-chart-5"),
        Colour("fill-chart-6"),
        Colour("fill-chart-7"),
        Colour("fill-chart-8"),
    ];
    /// Ordered categories (buckets, tiers), lowest first: one hue in steps,
    /// so the order reads in the colour.
    pub const RAMP: [Colour; 5] = [
        Colour("fill-chart-ramp-1"),
        Colour("fill-chart-ramp-2"),
        Colour("fill-chart-ramp-3"),
        Colour("fill-chart-ramp-4"),
        Colour("fill-chart-ramp-5"),
    ];
    /// What is not a series of its own, such as "other" or "none".
    pub const OTHER: Colour = Colour("fill-muted-foreground");
}

/// One named value: a segment of a `StackedBar`, a row of an `HBarList`.
#[derive(Clone, Debug)]
pub struct Part {
    pub label: String,
    pub value: Option<f64>,
    pub colour: Colour,
}

impl Part {
    pub fn new(label: impl Into<String>, value: Option<f64>, colour: Colour) -> Self {
        Part {
            label: label.into(),
            value,
            colour,
        }
    }
}

/// One series of a `BarSeries`: a value per bar, `None` where there is none.
#[derive(Clone, Debug)]
pub struct Series {
    pub label: String,
    pub values: Vec<Option<f64>>,
    pub colour: Colour,
}

impl Series {
    pub fn new(label: impl Into<String>, values: Vec<Option<f64>>, colour: Colour) -> Self {
        Series {
            label: label.into(),
            values,
            colour,
        }
    }
}

// ─── Values, scales and labels ──────────────────────────────────────────────

/// The value as a plottable magnitude: finite and not negative. Anything
/// else is no data.
fn known(v: Option<f64>) -> Option<f64> {
    v.filter(|x| x.is_finite() && *x >= 0.0)
}

/// A count with thousands separators, `18,342`: the charts' default format.
pub fn fmt_count(v: f64) -> String {
    let rounded = format!("{v:.0}");
    let (sign, digits) = match rounded.strip_prefix('-') {
        Some(d) => ("-", d),
        None => ("", rounded.as_str()),
    };
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, d) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(d);
    }
    format!("{sign}{grouped}")
}

/// An axis tick, compact (`750`, `1.5K`, `2.25M`) to fit beside the plot,
/// and exact for the round values a `Scale` picks.
fn fmt_tick(v: f64) -> String {
    let (n, unit) = [(1e9, "B"), (1e6, "M"), (1e3, "K")]
        .into_iter()
        .find(|(base, _)| v >= *base)
        .map_or((v, ""), |(base, unit)| (v / base, unit));
    let fixed = format!("{n:.2}");
    let trimmed = fixed.trim_end_matches('0').trim_end_matches('.');
    format!("{trimmed}{unit}")
}

/// A share as a whole percentage, where a share above zero never rounds
/// down to a fictitious `0%`.
fn fmt_share(share: f64) -> String {
    if share > 0.0 && share < 0.005 {
        "<1%".to_string()
    } else {
        format!("{:.0}%", share * 100.0)
    }
}

/// A share of the chart's width, as an SVG length.
fn pct(share: f64) -> String {
    format!("{:.2}%", share * 100.0)
}

/// A value axis from zero to `step * intervals`, ticked at every step.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Scale {
    step: f64,
    intervals: u32,
}

impl Scale {
    fn top(self) -> f64 {
        self.step * f64::from(self.intervals)
    }
}

/// A round axis over `max`, picked as d3's `tickIncrement` picks one: aim
/// for four intervals, take the step of 1, 2 or 5 times a power of ten
/// nearest that, and top the axis at the first step at or above `max`.
/// Whole-number data steps by whole numbers. A max of zero still gets an
/// axis, so a zero is drawn as a zero.
fn scale(max: f64, whole: bool) -> Scale {
    if max <= 0.0 {
        return Scale {
            step: 1.0,
            intervals: 1,
        };
    }
    let raw = max / 4.0;
    let power = 10f64.powf(raw.log10().floor());
    let error = raw / power;
    let nice = if error >= 50f64.sqrt() {
        10.0
    } else if error >= 10f64.sqrt() {
        5.0
    } else if error >= 2f64.sqrt() {
        2.0
    } else {
        1.0
    };
    let step = if whole {
        (nice * power).max(1.0)
    } else {
        nice * power
    };
    // Shaved before the ceiling, so float noise never adds an empty step.
    let intervals = (max / step * (1.0 - 1e-12)).ceil().max(1.0) as u32;
    Scale { step, intervals }
}

/// What a chart with nothing to draw shows instead: said in words, so a
/// missing value never reads as a zero.
fn no_data(label: &str) -> AnyView {
    let summary = format!("{label}: no data");
    view! {
        <p role="img" aria-label=summary class="text-sm font-normal text-muted-foreground">
            "no data"
        </p>
    }
    .into_any()
}

/// Each series' swatch and label, and its figure when it has one: identity
/// never rests on matching a colour alone.
fn legend(items: Vec<(Colour, String, Option<String>)>) -> impl IntoView {
    let items = items
        .into_iter()
        .map(|(colour, label, figure)| {
            view! {
                <li class="flex min-w-0 items-center gap-1.5">
                    <svg class="size-2.5 shrink-0">
                        <rect width="100%" height="100%" rx="2" class=colour.0 />
                    </svg>
                    <span class="wrap-break-word">{label}</span>
                    {figure.map(|f| view! { <span class="text-muted-foreground">{f}</span> })}
                </li>
            }
        })
        .collect_view();
    view! { <ul class="flex flex-wrap gap-x-4 gap-y-1 text-xs font-normal">{items}</ul> }
}

// ─── Meter ──────────────────────────────────────────────────────────────────

/// How full a meter is against its thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Level {
    Ok,
    Warn,
    Crit,
}

fn level(share: f64, warn: f64, crit: f64) -> Level {
    if share >= crit {
        Level::Crit
    } else if share >= warn {
        Level::Warn
    } else {
        Level::Ok
    }
}

/// Used against a limit: the figures and the share on a line, over a bar
/// whose fill turns to the warning colour and then the critical one as the
/// share crosses `warn` and `crit`, with a tick at each threshold short of
/// the limit. The bar spans its container, so meters in a column line up.
/// The level is in the tooltip and the accessible name in words; on screen
/// the tick and the printed share carry it as well as the colour. A missing
/// or zero limit gives no share, so the meter says "no data" rather than
/// draw one; an over-full meter is drawn full and its share printed as is.
#[component]
pub fn Meter(
    /// What is measured: the accessible name and the tooltip's lead.
    #[prop(into)]
    label: String,
    used: Option<f64>,
    limit: Option<f64>,
    /// The share of the limit at which the fill turns to warning.
    #[prop(default = 0.8)]
    warn: f64,
    /// The share at which it turns critical: by default the limit itself.
    #[prop(default = 1.0)]
    crit: f64,
    /// Writes the used and limit figures beside the share. Without it the
    /// meter shows the share alone, for a value that is one already, such
    /// as a utilisation.
    #[prop(optional)]
    fmt: Option<fn(f64) -> String>,
) -> impl IntoView {
    let used = known(used);
    let figures = fmt.map(|f| {
        let figure = |v: Option<f64>| v.map_or_else(|| "—".to_string(), f);
        format!("{} / {}", figure(used), figure(known(limit)))
    });
    let share = used
        .zip(known(limit).filter(|l| *l > 0.0))
        .map(|(u, l)| u / l);
    let Some(share) = share else {
        let text = figures.map_or_else(|| "no data".to_string(), |f| format!("{f} · no data"));
        let summary = format!("{label}: {text}");
        return view! {
            <p role="img" aria-label=summary class="text-sm font-normal text-muted-foreground tabular-nums">
                {text}
            </p>
        }
        .into_any();
    };

    let percent = fmt_share(share);
    let (fill, state) = match level(share, warn, crit) {
        Level::Ok => ("fill-chart-1", ""),
        Level::Warn => ("fill-chart-warn", ", warning"),
        Level::Crit => ("fill-chart-crit", ", critical"),
    };
    let summary = match &figures {
        Some(f) => format!("{label}: {f}, {percent}{state}"),
        None => format!("{label}: {percent}{state}"),
    };
    let tip = summary.clone();
    let filled = (share > 0.0).then(|| {
        let width = pct(share.min(1.0));
        view! { <rect width=width height="100%" rx="4" class=fill /> }
    });
    let ticks = [warn, crit]
        .into_iter()
        .filter(|t| *t > 0.0 && *t < 1.0)
        .map(|t| {
            let (x1, x2) = (pct(t), pct(t));
            view! { <line x1=x1 x2=x2 y1="0" y2="100%" stroke-width="1" class="stroke-muted-foreground" /> }
        })
        .collect_view();
    view! {
        <div role="img" aria-label=summary class="min-w-0 space-y-1 font-normal">
            <div class="flex flex-wrap items-baseline gap-x-2 text-sm tabular-nums">
                {figures.map(|f| view! { <span>{f}</span> })}
                <span class="ml-auto">{percent}</span>
            </div>
            <svg class="h-2 w-full">
                <title>{tip}</title>
                <rect width="100%" height="100%" rx="4" class="fill-secondary" />
                {filled}
                {ticks}
            </svg>
        </div>
    }
    .into_any()
}

// ─── StackedBar ─────────────────────────────────────────────────────────────

/// Each part above zero as (part, start, width), both shares of the whole;
/// `None` while any part is unknown, since the whole is then unknown too.
fn shares(values: &[Option<f64>]) -> Option<Vec<(usize, f64, f64)>> {
    let values: Vec<f64> = values.iter().map(|v| known(*v)).collect::<Option<_>>()?;
    let total: f64 = values.iter().sum();
    let mut start = 0.0;
    Some(
        values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                let width = if total > 0.0 { v / total } else { 0.0 };
                let segment = (width > 0.0).then_some((i, start, width));
                start += width;
                segment
            })
            .collect(),
    )
}

/// A part-to-whole mix: one bar split by share, then a legend with each
/// part's value and share. A part with no data leaves the whole unknown, so
/// the bar is not drawn and the legend gives the values there are. Parts
/// that are all zero draw an empty bar.
#[component]
pub fn StackedBar(
    /// What the whole is: the accessible name.
    #[prop(into)]
    label: String,
    /// The parts in their fixed display order, zeros included, so each keeps
    /// its colour whatever the data.
    parts: Vec<Part>,
    #[prop(default = fmt_count)] fmt: fn(f64) -> String,
) -> impl IntoView {
    if parts.iter().all(|p| known(p.value).is_none()) {
        return no_data(&label);
    }
    let values: Vec<Option<f64>> = parts.iter().map(|p| p.value).collect();
    let segments = shares(&values);
    let total: f64 = values.iter().filter_map(|v| known(*v)).sum();
    let whole = segments.is_some() && total > 0.0;
    let figure = |v: Option<f64>| match known(v) {
        Some(v) if whole => format!("{} ({})", fmt(v), fmt_share(v / total)),
        Some(v) => fmt(v),
        None => "no data".to_string(),
    };
    let listed = parts
        .iter()
        .map(|p| format!("{} {}", p.label, figure(p.value)))
        .collect::<Vec<_>>()
        .join(", ");
    let summary = match segments {
        Some(_) => format!("{label}: {listed}; total {}", fmt(total)),
        None => format!("{label}: {listed}; the whole is unknown"),
    };
    let bar = match segments {
        Some(segments) if whole => {
            // Each part is its own clipping viewport; every part after the
            // first starts 2px in, the surface gap between neighbours.
            let marks = segments
                .into_iter()
                .map(|(i, start, width)| {
                    let p = &parts[i];
                    let inset = if start > 0.0 { "2" } else { "0" };
                    let tip = format!("{}: {}", p.label, figure(p.value));
                    view! {
                        <svg x=pct(start) width=pct(width)>
                            <rect x=inset width="100%" height="100%" class=p.colour.0>
                                <title>{tip}</title>
                            </rect>
                        </svg>
                    }
                })
                .collect_view();
            Either::Left(view! { <svg class="h-3 w-full rounded-full">{marks}</svg> })
        }
        Some(_) => Either::Right(Either::Left(view! {
            <svg class="h-3 w-full">
                <rect width="100%" height="100%" rx="6" class="fill-secondary">
                    <title>{format!("{label}: total 0")}</title>
                </rect>
            </svg>
        })),
        None => Either::Right(Either::Right(view! {
            <p class="text-sm text-muted-foreground">"no data for the whole"</p>
        })),
    };
    let items = parts
        .iter()
        .map(|p| (p.colour, p.label.clone(), Some(figure(p.value))))
        .collect();
    view! {
        <div role="img" aria-label=summary class="space-y-2 font-normal">
            {bar}
            {legend(items)}
        </div>
    }
    .into_any()
}

// ─── BarSeries ──────────────────────────────────────────────────────────────

/// Plot height.
const PLOT_H: f64 = 120.0;
/// Room above the plot for the top tick's label.
const PLOT_TOP: f64 = 6.0;
/// The baseline's y. Under it, an 18px band of x labels: 144px in all, the
/// `h-36` of both of the chart's SVGs.
const BASELINE: f64 = PLOT_TOP + PLOT_H;
/// The surface gap between stacked segments.
const GAP: f64 = 2.0;
/// A bar's width cap. Within it a bar takes 60% of its slot, so the
/// remaining air never falls below 40%.
const BAR_W: f64 = 24.0;

/// One bar of a `BarSeries`.
#[derive(Debug, PartialEq)]
enum Column {
    /// A series has no value here, so the bar's total is unknown.
    NoData,
    /// The stack, bottom up, as (series, top y, height) in pixels; the
    /// stack is empty for a zero.
    Bar(Vec<(usize, f64, f64)>),
}

/// The bars of `n` slots on a value axis topped at `top`. A segment's height
/// is its value's share of the axis; each segment above the lowest gives its
/// bottom 2px to the gap, so the stack's top stays exact, and a segment too
/// short to show past its gap is left out of the drawing (not the tooltip).
fn columns(n: usize, series: &[Series], top: f64) -> Vec<Column> {
    (0..n)
        .map(|i| {
            let values: Option<Vec<f64>> = series
                .iter()
                .map(|s| known(s.values.get(i).copied().flatten()))
                .collect();
            let Some(values) = values else {
                return Column::NoData;
            };
            let mut base = BASELINE;
            let mut stack = Vec::new();
            for (j, v) in values.into_iter().enumerate() {
                let h = v / top * PLOT_H;
                if h <= 0.0 {
                    continue;
                }
                let gap = if stack.is_empty() { 0.0 } else { GAP };
                base -= h;
                if h - gap > 0.0 {
                    stack.push((j, base, h - gap));
                }
            }
            Column::Bar(stack)
        })
        .collect()
}

/// A column segment, centred in its slot and `BAR_W` wide at most; the top
/// one is rounded, square below its rounding.
fn segment(y: f64, h: f64, colour: Colour, rounded: bool, tip: String) -> impl IntoView {
    let (y_text, h_text) = (format!("{y:.2}"), format!("{h:.2}"));
    let cover = (rounded && h > 4.0).then(|| {
        let (cy, ch) = (format!("{:.2}", y + 4.0), format!("{:.2}", h - 4.0));
        view! {
            <rect x="50%" y=cy width=BAR_W height=ch pointer-events="none" class=colour.0 />
        }
    });
    let rx = if rounded { "4" } else { "0" };
    view! {
        <rect x="50%" y=y_text width=BAR_W height=h_text rx=rx class=colour.0>
            <title>{tip}</title>
        </rect>
        {cover}
    }
}

/// Bars over a run of days or other slots, one value per slot from each
/// series, stacked in series order. The value axis starts at zero and is
/// ticked at round values; the first and last slots are labelled below it;
/// each segment and each slot has a tooltip with its figures. A slot where a
/// series has no value draws a ghost bar, since its total is unknown; a zero
/// draws nothing above the baseline.
#[component]
pub fn BarSeries(
    /// What is counted: the accessible name.
    #[prop(into)]
    label: String,
    /// The slots' labels, in order, such as days.
    x: Vec<String>,
    /// The series in their fixed display order, bottom of the stack first.
    series: Vec<Series>,
    #[prop(default = fmt_count)] fmt: fn(f64) -> String,
) -> impl IntoView {
    let n = x.len();
    let totals: Vec<Option<f64>> = (0..n)
        .map(|i| {
            series
                .iter()
                .map(|s| known(s.values.get(i).copied().flatten()))
                .sum()
        })
        .collect();
    if series.is_empty() || totals.iter().all(Option::is_none) {
        return no_data(&label);
    }
    let whole = series
        .iter()
        .flat_map(|s| s.values.iter().take(n))
        .filter_map(|v| known(*v))
        .all(|v| v.fract() == 0.0);
    let max = totals.iter().flatten().copied().fold(0.0, f64::max);
    let axis = scale(max, whole);
    let (top, intervals) = (axis.top(), axis.intervals);

    // On the pixel grid, so each 1px hairline stays one pixel.
    let tick_y = |i: u32| (PLOT_TOP + PLOT_H * (1.0 - f64::from(i) / f64::from(intervals))).round();
    let tick_labels = (0..=intervals)
        .map(|i| {
            let (y, text) = (format!("{:.1}", tick_y(i)), fmt_tick(axis.step * f64::from(i)));
            view! {
                <text x="34" y=y text-anchor="end" dominant-baseline="middle" class="fill-muted-foreground text-[11px] tabular-nums">
                    {text}
                </text>
            }
        })
        .collect_view();
    let gridlines = (1..=intervals)
        .map(|i| {
            let (y1, y2) = (
                format!("{:.1}", tick_y(i) + 0.5),
                format!("{:.1}", tick_y(i) + 0.5),
            );
            view! { <line x1="0" x2="100%" y1=y1 y2=y2 stroke-width="1" class="stroke-border" /> }
        })
        .collect_view();

    let slot = 1.0 / n as f64;
    let figure = |v: Option<f64>| known(v).map_or_else(|| "no data".to_string(), fmt);
    let bars = columns(n, &series, top)
        .into_iter()
        .enumerate()
        .map(|(i, column)| {
            let figures = series
                .iter()
                .map(|s| format!("{} {}", s.label, figure(s.values.get(i).copied().flatten())))
                .collect::<Vec<_>>()
                .join(", ");
            let slot_tip = match totals[i] {
                Some(t) if series.len() > 1 => format!("{}: {figures}; total {}", x[i], fmt(t)),
                _ => format!("{}: {figures}", x[i]),
            };
            let marks = match column {
                Column::NoData => {
                    let tip = format!("{}: no data", x[i]);
                    Either::Left(view! {
                        <rect x="50%" y=PLOT_TOP width=BAR_W height=PLOT_H rx="4" class="fill-secondary">
                            <title>{tip}</title>
                        </rect>
                    })
                }
                Column::Bar(stack) => {
                    let last = stack.len().saturating_sub(1);
                    Either::Right(
                        stack
                            .into_iter()
                            .enumerate()
                            .map(|(k, (j, y, h))| {
                                let s = &series[j];
                                let tip = format!(
                                    "{}: {} {}",
                                    x[i],
                                    s.label,
                                    figure(s.values.get(i).copied().flatten())
                                );
                                segment(y, h, s.colour, k == last, tip)
                            })
                            .collect_view(),
                    )
                }
            };
            view! {
                <svg x=pct(slot * i as f64) width=pct(slot)>
                    <rect y=PLOT_TOP width="100%" height=PLOT_H class="fill-transparent">
                        <title>{slot_tip}</title>
                    </rect>
                    // The bar's room: 60% of the slot, centred. A bar wider
                    // than its room is clipped to it.
                    <svg x="20%" width="60%">
                        <g transform=format!("translate({} 0)", -BAR_W / 2.0)>{marks}</g>
                    </svg>
                </svg>
            }
        })
        .collect_view();

    let first = x.first().cloned().unwrap_or_default();
    let last = (n > 1).then(|| x[n - 1].clone());
    let label_y = format!("{:.1}", BASELINE + 14.0);
    let baseline_y = format!("{:.1}", BASELINE + 0.5);
    let x_labels = view! {
        <text x="0" y=label_y.clone() class="fill-muted-foreground text-[11px]">{first.clone()}</text>
        {last.clone().map(|l| view! {
            <text x="100%" y=label_y text-anchor="end" class="fill-muted-foreground text-[11px]">{l}</text>
        })}
    };

    let peak = totals
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.map(|t| (i, t)))
        .fold(None, |best: Option<(usize, f64)>, (i, t)| match best {
            Some((_, b)) if b >= t => best,
            _ => Some((i, t)),
        });
    let missing = totals.iter().filter(|t| t.is_none()).count();
    let span = last.map_or_else(|| first.clone(), |l| format!("{first} to {l}"));
    let names = series
        .iter()
        .map(|s| s.label.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let mut summary = format!("{label}, {span}: {names}");
    if let Some((i, t)) = peak {
        summary.push_str(&format!("; highest {} on {}", fmt(t), x[i]));
    }
    if let Some(t) = totals[n - 1] {
        summary.push_str(&format!("; latest {} on {}", fmt(t), x[n - 1]));
    }
    if missing > 0 {
        summary.push_str(&format!("; no data on {missing} of {n}"));
    }
    let key = (series.len() > 1).then(|| {
        legend(
            series
                .iter()
                .map(|s| (s.colour, s.label.clone(), None))
                .collect(),
        )
    });

    view! {
        <div role="img" aria-label=summary class="space-y-2 font-normal">
            <div class="flex">
                <svg class="h-36 w-10 shrink-0">{tick_labels}</svg>
                <svg class="h-36 min-w-0 flex-1">
                    {gridlines}
                    <line x1="0" x2="100%" y1=baseline_y.clone() y2=baseline_y stroke-width="1" class="stroke-muted-foreground/50" />
                    {bars}
                    {x_labels}
                </svg>
            </div>
            {key}
        </div>
    }
    .into_any()
}

// ─── Sparkline ──────────────────────────────────────────────────────────────

/// The sparkline's height, and the room it keeps at the top and bottom for
/// the end dot: its 4px radius and 2px ring, less a pixel of overlap.
const SPARK_H: f64 = 32.0;
const SPARK_PAD: f64 = 5.0;

/// A value's height in the sparkline, on an axis from zero to `top`.
fn spark_y(v: f64, top: f64) -> f64 {
    SPARK_PAD + (1.0 - v / top) * (SPARK_H - 2.0 * SPARK_PAD)
}

/// A point's position across the sparkline, as a share of its width.
fn spark_x(i: usize, n: usize) -> f64 {
    if n <= 1 {
        0.5
    } else {
        i as f64 / (n - 1) as f64
    }
}

/// The runs of consecutive values with data, as (index, value): a gap in
/// the data is a gap in the line.
fn runs(values: &[Option<f64>]) -> Vec<Vec<(usize, f64)>> {
    let mut out: Vec<Vec<(usize, f64)>> = Vec::new();
    let mut open = false;
    for (i, v) in values.iter().enumerate() {
        match known(*v) {
            Some(v) if open => out.last_mut().into_iter().for_each(|r| r.push((i, v))),
            Some(v) => {
                out.push(vec![(i, v)]);
                open = true;
            }
            None => open = false,
        }
    }
    out
}

/// A trend, small enough to sit in a `Stat` under its value: a line from
/// zero to the highest value, broken where there is no data, and the latest
/// value marked in the accent colour when it has one. Each point has a
/// tooltip over its slice of the width.
#[component]
pub fn Sparkline(
    /// What the trend is of: the accessible name.
    #[prop(into)]
    label: String,
    /// One value per point, oldest first.
    values: Vec<Option<f64>>,
    /// The points' labels, such as days.
    x: Vec<String>,
    #[prop(default = fmt_count)] fmt: fn(f64) -> String,
) -> impl IntoView {
    let n = values.len();
    let runs = runs(&values);
    if runs.is_empty() {
        return no_data(&label);
    }
    let point = |i: usize| {
        x.get(i)
            .cloned()
            .unwrap_or_else(|| format!("point {}", i + 1))
    };
    let figure = |v: Option<f64>| known(v).map_or_else(|| "no data".to_string(), fmt);
    let found = runs.iter().flatten().map(|(_, v)| *v);
    let max = found.clone().fold(0.0, f64::max);
    let min = found.fold(f64::INFINITY, f64::min);
    let top = if max > 0.0 { max } else { 1.0 };

    let lines = runs
        .iter()
        .filter(|r| r.len() > 1)
        .map(|r| {
            let points = r
                .iter()
                .map(|(i, v)| format!("{:.2},{:.2}", spark_x(*i, n) * 100.0, spark_y(*v, top)))
                .collect::<Vec<_>>()
                .join(" ");
            view! {
                <polyline points=points stroke-width="2" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke" class="fill-none stroke-muted-foreground" />
            }
        })
        .collect_view();
    let latest = known(values.last().copied().flatten()).map(|v| (n - 1, v));
    let dots = runs
        .iter()
        .filter(|r| r.len() == 1)
        .flatten()
        .copied()
        .filter(|p| Some(*p) != latest)
        .map(|(i, v)| (i, v, false))
        .chain(latest.map(|(i, v)| (i, v, true)))
        .map(|(i, v, accent)| {
            let (cx, cy) = (pct(spark_x(i, n)), format!("{:.2}", spark_y(v, top)));
            let tip = format!("{}: {}", point(i), fmt(v));
            if accent {
                Either::Left(view! {
                    <circle cx=cx cy=cy r="4" stroke-width="2" class="fill-chart-1 stroke-card"><title>{tip}</title></circle>
                })
            } else {
                Either::Right(view! {
                    <circle cx=cx cy=cy r="2" class="fill-muted-foreground"><title>{tip}</title></circle>
                })
            }
        })
        .collect_view();
    // Each point's tooltip covers the slice of width nearest to it.
    let slices = (0..n)
        .map(|i| {
            let half = if n <= 1 { 0.5 } else { 0.5 / (n - 1) as f64 };
            let from = (spark_x(i, n) - half).max(0.0);
            let to = (spark_x(i, n) + half).min(1.0);
            let tip = format!("{}: {}", point(i), figure(values[i]));
            view! {
                <rect x=pct(from) width=pct(to - from) height="100%" class="fill-transparent">
                    <title>{tip}</title>
                </rect>
            }
        })
        .collect_view();

    let mut summary = format!("{label}: {n} points");
    if let (Some(a), Some(b)) = (x.first(), x.get(n - 1)) {
        summary.push_str(&format!(" from {a} to {b}"));
    }
    summary.push_str(&format!(
        "; latest {}; highest {}; lowest {}",
        figure(values.last().copied().flatten()),
        fmt(max),
        fmt(min)
    ));
    let missing = values.iter().filter(|v| known(**v).is_none()).count();
    if missing > 0 {
        summary.push_str(&format!("; no data for {missing}"));
    }
    let view_box = format!("0 0 100 {SPARK_H}");
    view! {
        <div role="img" aria-label=summary class="mt-2">
            <svg class="h-8 w-full overflow-visible">
                {slices}
                // Stretched across, never up: the viewBox's height is the
                // SVG's, and the stroke ignores the stretch.
                <svg width="100%" height="100%" viewBox=view_box preserveAspectRatio="none" pointer-events="none" class="overflow-visible">
                    {lines}
                </svg>
                {dots}
            </svg>
        </div>
    }
    .into_any()
}

// ─── HBarList ───────────────────────────────────────────────────────────────

/// A ranking: one row per entry, in the order given, with its label and
/// value printed above a bar measured against the largest value. A row
/// with no data says so and draws no bar; a zero draws no bar.
#[component]
pub fn HBarList(
    /// What is ranked: the accessible name.
    #[prop(into)]
    label: String,
    /// The rows, largest first; an `OTHER` row for the rest goes last.
    rows: Vec<Part>,
    #[prop(default = fmt_count)] fmt: fn(f64) -> String,
) -> impl IntoView {
    if rows.iter().all(|r| known(r.value).is_none()) {
        return no_data(&label);
    }
    let top = rows
        .iter()
        .filter_map(|r| known(r.value))
        .fold(0.0, f64::max);
    let figure = |v: Option<f64>| known(v).map_or_else(|| "no data".to_string(), fmt);
    let summary = format!(
        "{label}: {}",
        rows.iter()
            .map(|r| format!("{} {}", r.label, figure(r.value)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let items = rows
        .into_iter()
        .map(|r| {
            let value = figure(r.value);
            let tip = format!("{}: {value}", r.label);
            let bar = known(r.value)
                .filter(|v| *v > 0.0 && top > 0.0)
                .map(|v| {
                    // Rounded at the value's end, square at the baseline.
                    let (whole, half) = (pct(v / top), pct(v / top / 2.0));
                    view! {
                        <rect width=whole height="100%" rx="4" class=r.colour.0><title>{tip}</title></rect>
                        <rect width=half height="100%" pointer-events="none" class=r.colour.0 />
                    }
                });
            let full = r.label.clone();
            view! {
                <li class="space-y-1">
                    <div class="flex items-baseline justify-between gap-3 text-sm">
                        <span class="min-w-0 wrap-break-word line-clamp-2" title=full>{r.label}</span>
                        <span class="shrink-0 tabular-nums">{value}</span>
                    </div>
                    <svg class="h-2 w-full">{bar}</svg>
                </li>
            }
        })
        .collect_view();
    view! { <ol role="img" aria-label=summary class="space-y-3 font-normal">{items}</ol> }
        .into_any()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn html(v: impl IntoView) -> String {
        v.into_view().to_html()
    }

    // ─── Scales, figures and geometry ───────────────────────────────────────

    #[test]
    fn the_scale_steps_round_and_tops_out_at_the_first_step_past_the_max() {
        let s = |step, intervals| Scale { step, intervals };
        assert_eq!(scale(137.0, true), s(50.0, 3));
        assert_eq!(scale(229.0, true), s(50.0, 5));
        assert_eq!(scale(818.0, true), s(200.0, 5));
        assert_eq!(scale(1000.0, true), s(200.0, 5));
        assert_eq!(scale(41.0, true), s(10.0, 5));
        assert_eq!(scale(10.0, true), s(2.0, 5));
        assert_eq!(scale(3.0, true), s(1.0, 3));
        assert_eq!(scale(1.0, true), s(1.0, 1));
        assert_eq!(scale(2.2e7, true), s(5e6, 5));
        assert_eq!(scale(0.62, false).intervals, 4);
        assert!((scale(0.62, false).top() - 0.8).abs() < 1e-12);
    }

    /// A count axis never offers half an item, every tick holds the data,
    /// and the gridlines stay few enough to read.
    #[test]
    fn whole_number_data_gets_whole_ticks() {
        for max in 1..=5000u32 {
            let axis = scale(f64::from(max), true);
            assert!(axis.top() >= f64::from(max), "{max}: {axis:?}");
            assert!(
                axis.top() - axis.step < f64::from(max),
                "{max}: an empty step on top: {axis:?}"
            );
            assert_eq!(axis.step.fract(), 0.0, "{max}: {axis:?}");
            assert!((1..=7).contains(&axis.intervals), "{max}: {axis:?}");
        }
    }

    /// Nothing to show still draws an axis, so the bars at zero stand on it.
    #[test]
    fn a_zero_max_still_gets_an_axis() {
        assert_eq!(
            scale(0.0, true),
            Scale {
                step: 1.0,
                intervals: 1
            }
        );
    }

    #[test]
    fn figures_are_written_in_full_and_ticks_compactly() {
        for (v, want) in [
            (0.0, "0"),
            (999.0, "999"),
            (1000.0, "1,000"),
            (18342.0, "18,342"),
            (1234567.4, "1,234,567"),
            (-1234.0, "-1,234"),
        ] {
            assert_eq!(fmt_count(v), want, "{v}");
        }
        for (v, want) in [
            (0.0, "0"),
            (0.25, "0.25"),
            (150.0, "150"),
            (1500.0, "1.5K"),
            (2.25e6, "2.25M"),
            (1e9, "1B"),
        ] {
            assert_eq!(fmt_tick(v), want, "{v}");
        }
        // A sliver of a whole is never printed as nothing.
        for (v, want) in [(0.0, "0%"), (0.004, "<1%"), (0.5, "50%"), (1.3, "130%")] {
            assert_eq!(fmt_share(v), want, "{v}");
        }
    }

    #[test]
    fn missing_infinite_nan_and_negative_values_are_not_known() {
        assert_eq!(known(Some(0.0)), Some(0.0));
        assert_eq!(known(Some(2.5)), Some(2.5));
        for v in [None, Some(f64::NAN), Some(f64::INFINITY), Some(-1.0)] {
            assert_eq!(known(v), None, "{v:?}");
        }
    }

    #[test]
    fn stacked_shares_cover_the_whole_in_order() {
        assert_eq!(
            shares(&[Some(1.0), Some(0.0), Some(3.0)]),
            Some(vec![(0, 0.0, 0.25), (2, 0.25, 0.75)])
        );
        assert_eq!(shares(&[Some(0.0), Some(0.0)]), Some(vec![]));
        assert_eq!(shares(&[Some(1.0), None]), None);
        assert_eq!(shares(&[Some(1.0), Some(f64::NAN)]), None);
    }

    fn series(values: &[&[Option<f64>]]) -> Vec<Series> {
        values
            .iter()
            .enumerate()
            .map(|(i, v)| Series::new(format!("s{i}"), v.to_vec(), Colour::SERIES[i]))
            .collect()
    }

    /// Heights are shares of the axis, stacked from the baseline in series
    /// order; the upper segment gives its bottom 2px to the gap, so the
    /// stack's top lands exactly on its total.
    #[test]
    fn columns_stack_up_from_the_baseline() {
        let s = series(&[&[Some(30.0)], &[Some(60.0)]]);
        let top_y = BASELINE - 90.0;
        assert_eq!(
            columns(1, &s, 120.0),
            vec![Column::Bar(vec![
                (0, BASELINE - 30.0, 30.0),
                (1, top_y, 60.0 - GAP)
            ])]
        );
    }

    #[test]
    fn a_missing_value_leaves_its_column_unknown_and_a_zero_empty() {
        let s = series(&[
            &[Some(1.0), Some(0.0), None, Some(f64::NAN)],
            &[Some(1.0), Some(0.0), Some(1.0), Some(1.0)],
        ]);
        let cols = columns(5, &s, 10.0);
        assert!(matches!(&cols[0], Column::Bar(stack) if stack.len() == 2));
        assert_eq!(cols[1], Column::Bar(vec![]));
        assert_eq!(cols[2], Column::NoData);
        assert_eq!(cols[3], Column::NoData);
        // Past the end of the values: no data, not zero.
        assert_eq!(cols[4], Column::NoData);
    }

    #[test]
    fn a_segment_too_short_for_its_gap_is_left_out_of_the_drawing() {
        // 1 in 120px is 1px, under the 2px gap; the one above still gaps.
        let s = series(&[&[Some(60.0)], &[Some(1.0)], &[Some(30.0)]]);
        let Column::Bar(stack) = &columns(1, &s, 120.0)[0] else {
            panic!("a bar");
        };
        let drawn: Vec<usize> = stack.iter().map(|(j, _, _)| *j).collect();
        assert_eq!(drawn, [0, 2]);
        assert_eq!(stack[1], (2, BASELINE - 91.0, 30.0 - GAP));
    }

    #[test]
    fn a_meter_turns_at_its_thresholds() {
        assert_eq!(level(0.79, 0.8, 1.0), Level::Ok);
        assert_eq!(level(0.8, 0.8, 1.0), Level::Warn);
        assert_eq!(level(0.99, 0.8, 1.0), Level::Warn);
        assert_eq!(level(1.0, 0.8, 1.0), Level::Crit);
        assert_eq!(level(1.3, 0.8, 1.0), Level::Crit);
    }

    #[test]
    fn a_sparkline_runs_from_zero_at_the_bottom_to_the_top() {
        assert_eq!(spark_y(0.0, 8.0), SPARK_H - SPARK_PAD);
        assert_eq!(spark_y(8.0, 8.0), SPARK_PAD);
        assert_eq!(spark_x(0, 5), 0.0);
        assert_eq!(spark_x(4, 5), 1.0);
        assert_eq!(spark_x(0, 1), 0.5);
        assert_eq!(
            runs(&[
                Some(1.0),
                None,
                Some(2.0),
                Some(3.0),
                Some(f64::NAN),
                Some(4.0)
            ]),
            vec![vec![(0, 1.0)], vec![(2, 2.0), (3, 3.0)], vec![(5, 4.0)]]
        );
    }

    // ─── Meter ──────────────────────────────────────────────────────────────

    #[test]
    fn a_meter_fills_its_share_and_prints_the_figures() {
        let out = html(view! {
            <Meter label="searches" used=Some(188.0) limit=Some(200.0) fmt=fmt_count />
        });
        assert!(out.contains(r#"width="94.00%""#), "{out}");
        assert!(out.contains("fill-chart-warn"), "{out}");
        assert!(
            out.contains(r#"<span>188 / 200</span><span class="ml-auto">94%</span>"#),
            "{out}"
        );
        assert!(
            out.contains(r#"aria-label="searches: 188 / 200, 94%, warning""#),
            "{out}"
        );
        assert!(
            out.contains("<title>searches: 188 / 200, 94%, warning</title>"),
            "{out}"
        );
        // A tick at the warning threshold; none at the limit, the end.
        assert!(out.contains(r#"x1="80.00%""#), "{out}");
        assert_eq!(out.matches("<line").count(), 1, "{out}");
    }

    #[test]
    fn a_meter_under_its_threshold_wears_the_accent_and_over_its_limit_is_full() {
        let ok =
            html(view! { <Meter label="adds" used=Some(12.0) limit=Some(40.0) fmt=fmt_count /> });
        assert!(ok.contains("fill-chart-1"), "{ok}");
        assert!(ok.contains(r#"aria-label="adds: 12 / 40, 30%""#), "{ok}");
        let over = html(
            view! { <Meter label="budget" used=Some(260.0) limit=Some(200.0) fmt=fmt_count /> },
        );
        assert!(over.contains(r#"width="100.00%""#), "{over}");
        assert!(over.contains("fill-chart-crit"), "{over}");
        assert!(
            over.contains(r#"<span class="ml-auto">130%</span>"#),
            "{over}"
        );
        assert!(over.contains("260 / 200, 130%, critical"), "{over}");
    }

    /// Without a format the meter is a share already, such as utilisation,
    /// and prints the share alone; thresholds may be set per meter.
    #[test]
    fn a_meter_without_a_format_prints_its_share() {
        let out =
            html(view! { <Meter label="5h" used=Some(0.62) limit=Some(1.0) warn=0.5 crit=0.9 /> });
        assert!(out.contains(r#"<span class="ml-auto">62%</span>"#), "{out}");
        assert!(out.contains(r#"aria-label="5h: 62%, warning""#), "{out}");
        assert!(out.contains("fill-chart-warn"), "{out}");
        assert!(
            out.contains(r#"x1="50.00%""#) && out.contains(r#"x1="90.00%""#),
            "{out}"
        );
    }

    /// A zero is a zero: an empty bar, not "no data".
    #[test]
    fn a_meter_at_zero_is_empty() {
        let out =
            html(view! { <Meter label="adds" used=Some(0.0) limit=Some(40.0) fmt=fmt_count /> });
        assert!(
            out.contains(r#"<span>0 / 40</span><span class="ml-auto">0%</span>"#),
            "{out}"
        );
        assert!(!out.contains("fill-chart-1"), "no fill: {out}");
        assert!(!out.contains("no data"), "{out}");
    }

    /// No share without a used figure and a limit above zero: no bar, and
    /// never a fictitious empty one.
    #[test]
    fn a_meter_without_a_share_says_no_data() {
        let cases = [
            (None, Some(40.0)),
            (Some(f64::NAN), Some(40.0)),
            (Some(-1.0), Some(40.0)),
            (Some(f64::INFINITY), Some(40.0)),
            (Some(4.0), None),
            (Some(4.0), Some(f64::NAN)),
            (Some(0.0), Some(0.0)),
        ];
        for (used, limit) in cases {
            let out = html(view! { <Meter label="cap" used=used limit=limit /> });
            assert!(out.contains(">no data</p>"), "{used:?}/{limit:?}: {out}");
            assert!(!out.contains("<svg"), "{used:?}/{limit:?}: {out}");
        }
        let out =
            html(view! { <Meter label="cap" used=Some(0.0) limit=Some(0.0) fmt=fmt_count /> });
        assert!(out.contains(">0 / 0 · no data</p>"), "{out}");
        let out = html(view! { <Meter label="cap" used=Some(4.0) limit=None fmt=fmt_count /> });
        assert!(
            out.contains(r#"aria-label="cap: 4 / — · no data""#),
            "{out}"
        );
    }

    // ─── StackedBar ─────────────────────────────────────────────────────────

    fn verdicts(values: [Option<f64>; 3]) -> Vec<Part> {
        ["contradiction", "supersession", "coexist"]
            .into_iter()
            .zip(values)
            .enumerate()
            .map(|(i, (label, v))| Part::new(label, v, Colour::SERIES[i]))
            .collect()
    }

    #[test]
    fn a_stacked_bar_splits_by_share_with_gaps_between_parts() {
        let out = html(view! {
            <StackedBar label="Open" parts=verdicts([Some(30.0), Some(0.0), Some(10.0)]) />
        });
        // Contradiction from 0 for 75%, coexist from 75% for 25%, 2px in.
        assert!(
            out.contains(r#"<svg x="0.00%" width="75.00%"><rect x="0""#),
            "{out}"
        );
        assert!(
            out.contains(r#"<svg x="75.00%" width="25.00%"><rect x="2""#),
            "{out}"
        );
        assert_eq!(
            out.matches("<title>").count(),
            2,
            "a tooltip per drawn part: {out}"
        );
        assert!(out.contains("<title>coexist: 10 (25%)</title>"), "{out}");
        // The zero part keeps its place in the legend, and its colour.
        assert!(out.contains(">supersession</span>"), "{out}");
        assert!(out.contains(">0 (0%)</span>"), "{out}");
        assert!(
            out.contains(r#"aria-label="Open: contradiction 30 (75%), supersession 0 (0%), coexist 10 (25%); total 40""#),
            "{out}"
        );
    }

    #[test]
    fn a_stacked_bar_with_a_part_missing_draws_no_whole() {
        let out = html(
            view! { <StackedBar label="Open" parts=verdicts([Some(30.0), None, Some(10.0)]) /> },
        );
        assert!(!out.contains("<svg x="), "no segments: {out}");
        assert!(out.contains("no data for the whole"), "{out}");
        // The values there are, without shares of an unknown whole.
        assert!(out.contains(">30</span>"), "{out}");
        assert!(out.contains(">no data</span>"), "{out}");
        assert!(out.contains("the whole is unknown"), "{out}");
    }

    #[test]
    fn a_stacked_bar_of_zeros_is_an_empty_bar() {
        let out = html(view! { <StackedBar label="Resolved" parts=verdicts([Some(0.0); 3]) /> });
        assert!(out.contains("<title>Resolved: total 0</title>"), "{out}");
        assert!(!out.contains("no data"), "{out}");
        assert_eq!(out.matches(">0</span>").count(), 3, "{out}");
    }

    #[test]
    fn a_stacked_bar_with_nothing_known_says_no_data() {
        for parts in [vec![], verdicts([None, Some(f64::NAN), Some(-2.0)])] {
            let out = html(view! { <StackedBar label="Open" parts=parts /> });
            assert!(out.contains(r#"aria-label="Open: no data""#), "{out}");
            assert!(!out.contains("<svg"), "{out}");
        }
    }

    // ─── BarSeries ──────────────────────────────────────────────────────────

    fn days(n: usize) -> Vec<String> {
        (1..=n).map(|d| format!("10-{d:02}")).collect()
    }

    #[test]
    fn bar_series_slots_ticks_and_labels() {
        let s = series(&[
            &[Some(30.0), Some(10.0), None],
            &[Some(60.0), Some(0.0), Some(5.0)],
        ]);
        let out = html(view! { <BarSeries label="Judged" x=days(3) series=s /> });
        for x in ["0.00%", "33.33%", "66.67%"] {
            assert!(
                out.contains(&format!(r#"<svg x="{x}" width="33.33%">"#)),
                "{x}: {out}"
            );
        }
        // 90 at most: an axis of 100 in one interval, ticked 0 and 100.
        assert!(
            out.contains(">0</text>") && out.contains(">100</text>"),
            "{out}"
        );
        assert!(
            out.contains(">10-01</text>") && out.contains(">10-03</text>"),
            "{out}"
        );
        assert!(
            !out.contains(">10-02</text>"),
            "only the ends are labelled: {out}"
        );
        // The third day's first series is missing: a ghost bar, so said.
        assert!(out.contains("<title>10-03: no data</title>"), "{out}");
        assert!(
            out.contains("<title>10-01: s0 30, s1 60; total 90</title>"),
            "a slot's tooltip has every series: {out}"
        );
        assert!(out.contains("<title>10-01: s1 60</title>"), "{out}");
        // Two series: a legend.
        assert!(
            out.contains(">s0</span>") && out.contains(">s1</span>"),
            "{out}"
        );
        assert!(
            out.contains(r#"aria-label="Judged, 10-01 to 10-03: s0, s1; highest 90 on 10-01; no data on 1 of 3""#),
            "{out}"
        );
    }

    /// The top segment is rounded and squared below its rounding, so only
    /// the data end is round; the one below it is square.
    #[test]
    fn only_the_top_of_a_stack_is_rounded() {
        let s = series(&[&[Some(30.0)], &[Some(60.0)]]);
        let out = html(view! { <BarSeries label="x" x=days(1) series=s /> });
        assert_eq!(out.matches(r#"rx="4""#).count(), 1, "{out}");
        assert_eq!(out.matches(r#"pointer-events="none""#).count(), 1, "{out}");
    }

    #[test]
    fn a_single_series_has_no_legend_and_reports_its_latest() {
        let s = series(&[&[Some(3.0), Some(7.0)]]);
        let out = html(view! { <BarSeries label="Admitted" x=days(2) series=s /> });
        assert!(!out.contains("<ul"), "{out}");
        assert!(out.contains("<title>10-02: s0 7</title>"), "{out}");
        assert!(out.contains("; latest 7 on 10-02"), "{out}");
    }

    #[test]
    fn bar_series_of_zeros_stand_on_an_axis() {
        let s = series(&[&[Some(0.0), Some(0.0)]]);
        let out = html(view! { <BarSeries label="Admitted" x=days(2) series=s /> });
        assert!(!out.contains("no data"), "{out}");
        assert!(!out.contains(r#"rx="4""#), "no bars: {out}");
        assert!(out.contains(">1</text>"), "{out}");
    }

    #[test]
    fn bar_series_with_nothing_known_say_no_data() {
        let cases = [
            (days(2), series(&[&[None, Some(f64::NAN)]])),
            (vec![], series(&[&[Some(1.0)]])),
            (days(2), vec![]),
        ];
        for (x, s) in cases {
            let out = html(view! { <BarSeries label="Judged" x=x series=s /> });
            assert!(out.contains(r#"aria-label="Judged: no data""#), "{out}");
        }
    }

    // ─── Sparkline ──────────────────────────────────────────────────────────

    #[test]
    fn a_sparkline_breaks_at_gaps_and_marks_the_latest() {
        let values = vec![Some(2.0), Some(4.0), None, Some(8.0), Some(6.0)];
        let out = html(view! { <Sparkline label="Burn" values=values x=days(5) /> });
        // Two runs, one line each; zero at the bottom, the highest at the top.
        assert_eq!(out.matches("<polyline").count(), 2, "{out}");
        assert!(out.contains(r#"points="0.00,21.50 25.00,16.00""#), "{out}");
        assert!(out.contains(r#"points="75.00,5.00 100.00,10.50""#), "{out}");
        assert!(
            out.contains(r#"cx="100.00%""#) && out.contains("fill-chart-1"),
            "{out}"
        );
        assert!(out.contains("<title>10-03: no data</title>"), "{out}");
        assert_eq!(
            out.matches("<title>").count(),
            6,
            "five points and the dot: {out}"
        );
        assert!(
            out.contains(r#"aria-label="Burn: 5 points from 10-01 to 10-05; latest 6; highest 8; lowest 2; no data for 1""#),
            "{out}"
        );
    }

    /// A latest value with no data gets no accent: the line just stops. A
    /// lone point between gaps is a dot, or it would not show.
    #[test]
    fn a_sparkline_ending_in_a_gap_has_no_accent() {
        let values = vec![Some(1.0), None, Some(3.0), None];
        let out = html(view! { <Sparkline label="Burn" values=values x=days(4) /> });
        assert!(!out.contains("fill-chart-1"), "{out}");
        assert!(!out.contains("<polyline"), "{out}");
        assert_eq!(out.matches(r#"r="2""#).count(), 2, "{out}");
        assert!(out.contains("latest no data"), "{out}");
    }

    #[test]
    fn a_sparkline_of_zeros_lies_along_the_bottom() {
        let out = html(view! { <Sparkline label="Burn" values=vec![Some(0.0); 3] x=days(3) /> });
        assert!(
            out.contains(r#"points="0.00,27.00 50.00,27.00 100.00,27.00""#),
            "{out}"
        );
        assert!(!out.contains("no data"), "{out}");
    }

    #[test]
    fn a_sparkline_with_nothing_known_says_no_data() {
        for values in [vec![], vec![None, Some(f64::NAN)]] {
            let out = html(view! { <Sparkline label="Burn" values=values x=days(2) /> });
            assert!(out.contains(r#"aria-label="Burn: no data""#), "{out}");
        }
    }

    // ─── HBarList ───────────────────────────────────────────────────────────

    #[test]
    fn a_ranking_measures_each_row_against_the_largest() {
        let rows = vec![
            Part::new("timeout", Some(40.0), Colour::SERIES[0]),
            Part::new("connect", Some(10.0), Colour::SERIES[0]),
            Part::new("refused", Some(0.0), Colour::SERIES[0]),
            Part::new("reset", None, Colour::SERIES[0]),
            Part::new("other", Some(20.0), Colour::OTHER),
        ];
        let out = html(view! { <HBarList label="Errors" rows=rows /> });
        assert!(out.contains(r#"<rect width="100.00%""#), "{out}");
        assert!(out.contains(r#"<rect width="25.00%""#), "{out}");
        assert!(
            out.contains(
                r#"<rect width="50.00%" height="100%" rx="4" class="fill-muted-foreground">"#
            ),
            "{out}"
        );
        assert_eq!(
            out.matches("<title>").count(),
            3,
            "a bar for each row above zero: {out}"
        );
        assert!(
            out.contains(">0</span>") && out.contains(">no data</span>"),
            "{out}"
        );
        assert!(
            out.contains(
                r#"aria-label="Errors: timeout 40, connect 10, refused 0, reset no data, other 20""#
            ),
            "{out}"
        );
    }

    #[test]
    fn a_ranking_with_nothing_known_says_no_data() {
        for rows in [vec![], vec![Part::new("x", None, Colour::SERIES[0])]] {
            let out = html(view! { <HBarList label="Errors" rows=rows /> });
            assert!(out.contains(r#"aria-label="Errors: no data""#), "{out}");
        }
    }

    // ─── Every chart ────────────────────────────────────────────────────────

    /// Every chart in every state the tests above draw, its labels `label`.
    fn every_chart(label: &str) -> Vec<String> {
        let parts = |values: [Option<f64>; 2]| {
            values
                .into_iter()
                .enumerate()
                .map(|(i, v)| Part::new(format!("{label} {i}"), v, Colour::SERIES[i]))
                .collect::<Vec<_>>()
        };
        let series = |values: [Option<f64>; 3]| {
            vec![
                Series::new(label, values.to_vec(), Colour::SERIES[0]),
                Series::new(label, vec![Some(1.0); 3], Colour::SERIES[1]),
            ]
        };
        let x = || vec![label.to_string(); 3];
        let l = || label.to_string();
        let mut out = vec![];
        for (used, limit) in [
            (Some(5.0), Some(10.0)),
            (Some(9.0), Some(10.0)),
            (Some(11.0), Some(10.0)),
            (Some(0.0), Some(10.0)),
            (None, Some(1.0)),
            (Some(1.0), Some(0.0)),
        ] {
            out.push(html(
                view! { <Meter label=l() used=used limit=limit fmt=fmt_count /> },
            ));
            out.push(html(view! { <Meter label=l() used=used limit=limit /> }));
        }
        for values in [
            [Some(1.0), Some(2.0)],
            [Some(0.0), Some(0.0)],
            [Some(1.0), None],
            [None, None],
        ] {
            out.push(html(view! { <StackedBar label=l() parts=parts(values) /> }));
            out.push(html(view! { <HBarList label=l() rows=parts(values) /> }));
        }
        for values in [[Some(1.0), None, Some(3.0)], [Some(0.0); 3], [None; 3]] {
            out.push(html(
                view! { <BarSeries label=l() x=x() series=series(values) /> },
            ));
            out.push(html(
                view! { <Sparkline label=l() values=values.to_vec() x=x() /> },
            ));
        }
        out
    }

    #[test]
    fn no_chart_writes_a_style_attribute_or_a_script() {
        for out in every_chart("x") {
            for forbidden in [
                concat!("sty", "le="),
                concat!("<sty", "le"),
                concat!("<scr", "ipt"),
            ] {
                assert!(!out.contains(forbidden), "{forbidden}: {out}");
            }
        }
    }

    /// One image per chart, named by a summary that starts with its label,
    /// so a screen reader hears each chart once, in words.
    #[test]
    fn every_chart_is_one_image_with_a_summary() {
        for out in every_chart("Pairs judged") {
            assert_eq!(out.matches(r#"role="img""#).count(), 1, "{out}");
            assert!(
                out.contains(r#"aria-label="Pairs judged: "#)
                    || out.contains(r#"aria-label="Pairs judged, "#),
                "{out}"
            );
        }
    }

    /// Labels come from upstream data: in a tooltip, a legend, an axis or
    /// the summary, markup in them stays text.
    #[test]
    fn every_chart_escapes_its_labels() {
        let hostile = r#"<script>alert("x")</script> & "q""#;
        for out in every_chart(hostile) {
            assert!(!out.contains("<script>"), "{out}");
            assert!(out.contains("&lt;script&gt;"), "{out}");
        }
    }
}
