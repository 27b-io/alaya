//! Golden-set evaluation of the contradiction judge (LAB-3283 AC-8).
//!
//! `#[ignore]` — needs a live Ālaya to fetch pair contents and a judge
//! endpoint. The fixture holds hashes and human labels only (memory content
//! is the memory of record and never lands in git).
//!
//! ```bash
//! ALAYA_URL=http://localhost:13001 ALAYA_API_KEY=... \
//! JUDGE_URL=http://localhost:18082 JUDGE_API_KEY=... JUDGE_MODEL=claude-haiku-4-5 \
//!   cargo test -p alaya-core --test golden_judge -- --ignored --nocapture
//! ```
//!
//! Prints the confusion matrix, per-class precision/recall, survivor
//! accuracy and token totals (price them at the current list rate; a price
//! table here would rot). It does NOT assert the quality bar: missing it
//! blocks Phase 2 (LAB-3285), not this test.

use std::collections::HashMap;

use futures::StreamExt;
use serde::Deserialize;

use alaya_backends::judge::JudgeClient;
use alaya_backends::{ContradictionJudge, Survivor};
use alaya_types::graph::Verdict;
use alaya_types::memory::Memory;

const FIXTURE: &str = include_str!("fixtures/contradiction_golden.json");
const CONCURRENCY: usize = 4;

#[derive(Deserialize)]
struct Fixture {
    pairs: Vec<GoldenPair>,
}

#[derive(Deserialize, Clone)]
struct GoldenPair {
    a: String,
    b: String,
    label: Verdict,
    #[serde(default)]
    survivor: Option<String>,
    #[serde(default)]
    source: String,
}

#[derive(Deserialize)]
struct MemoryEnvelope {
    found: bool,
    #[serde(default)]
    memory: Option<Memory>,
}

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("{key} must be set for the golden-set run"))
}

#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn golden_set_precision_recall() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("fixture parses");
    assert!(
        fixture.pairs.len() >= 50,
        "AC-8 requires ≥50 labelled pairs, fixture has {}",
        fixture.pairs.len()
    );
    // A class with no labelled sample cannot be measured; say so up front
    // rather than printing NaN and letting a reader miss it. Not a failure:
    // the golden set is grown by human ratification, and the other classes
    // are still worth measuring meanwhile.
    let unmeasured: Vec<&str> = Verdict::CLASSES
        .iter()
        .filter(|c| !fixture.pairs.iter().any(|p| p.label == **c))
        .map(|c| c.as_str())
        .collect();
    if !unmeasured.is_empty() {
        println!("\n!! UNMEASURED CLASSES (no labelled pairs in the fixture): {unmeasured:?}");
    }

    let alaya_url = env("ALAYA_URL");
    let alaya_key = env("ALAYA_API_KEY");
    let model = env("JUDGE_MODEL");
    let judge = JudgeClient::new(env("JUDGE_URL"), model.clone(), Some(env("JUDGE_API_KEY")))
        .expect("JUDGE_API_KEY rejected — must be a single line of visible ASCII");
    // Test harness: a transport failure is a failed run, so `expect` is the
    // right shape here; the timeout keeps a dead endpoint from hanging it.
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .expect("reqwest client");

    // Fetch every distinct memory once.
    let mut hashes: Vec<&str> = fixture
        .pairs
        .iter()
        .flat_map(|p| [p.a.as_str(), p.b.as_str()])
        .collect();
    hashes.sort();
    hashes.dedup();
    let mut memories: HashMap<String, Memory> = HashMap::new();
    let mut missing = 0;
    for h in hashes {
        let resp = http
            .get(format!("{alaya_url}/memories/{h}"))
            .bearer_auth(&alaya_key)
            .send()
            .await
            .expect("alaya reachable");
        let env: MemoryEnvelope = resp.json().await.expect("memory envelope");
        match env.memory {
            Some(m) if env.found => {
                memories.insert(h.to_string(), m);
            }
            _ => {
                missing += 1;
                eprintln!("missing memory {h} (deleted since labelling?)");
            }
        }
    }

    let judge = &judge;
    let memories = &memories;
    let results: Vec<(GoldenPair, Option<alaya_backends::Judgement>)> =
        futures::stream::iter(fixture.pairs.iter().cloned())
            .map(|p| async move {
                let (Some(a), Some(b)) = (memories.get(&p.a), memories.get(&p.b)) else {
                    return (p, None);
                };
                match judge.judge(a, b).await {
                    Ok(j) => (p, Some(j)),
                    Err(e) => {
                        eprintln!("unjudged {}..{}: {e}", &p.a[..8], &p.b[..8]);
                        (p, None)
                    }
                }
            })
            .buffer_unordered(CONCURRENCY)
            .collect()
            .await;

    // Confusion matrix: predicted (rows, incl. unjudged) × label (cols).
    let classes = Verdict::CLASSES;
    let idx = |v: Verdict| classes.iter().position(|c| *c == v).unwrap();
    let mut matrix = [[0usize; 4]; 5]; // row 4 = unjudged
    let (mut in_tok, mut out_tok) = (0u64, 0u64);
    let (mut surv_total, mut surv_correct) = (0usize, 0usize);
    let mut skipped = 0usize;
    for (p, j) in &results {
        if !memories.contains_key(&p.a) || !memories.contains_key(&p.b) {
            skipped += 1;
            continue;
        }
        match j {
            Some(j) => {
                matrix[idx(j.verdict)][idx(p.label)] += 1;
                in_tok += j.input_tokens;
                out_tok += j.output_tokens;
                if p.label == Verdict::Supersession && j.verdict == Verdict::Supersession {
                    surv_total += 1;
                    let predicted = j.survivor.map(|s| match s {
                        Survivor::A => p.a.as_str(),
                        Survivor::B => p.b.as_str(),
                    });
                    if predicted == p.survivor.as_deref() {
                        surv_correct += 1;
                    }
                }
            }
            None => matrix[4][idx(p.label)] += 1,
        }
    }

    println!(
        "\n== golden set: model={model} pairs={} judged={} skipped(missing memory)={skipped} missing_memories={missing}",
        results.len(),
        results.len() - skipped - matrix[4].iter().sum::<usize>()
    );
    println!(
        "{:<14}{}",
        "pred \\ label",
        classes
            .iter()
            .map(|c| format!("{:>15}", c.as_str()))
            .collect::<String>()
    );
    for (r, row) in matrix.iter().enumerate() {
        let name = if r < 4 {
            classes[r].as_str()
        } else {
            Verdict::UNJUDGED
        };
        println!(
            "{name:<14}{}",
            row.iter().map(|n| format!("{n:>15}")).collect::<String>()
        );
    }
    println!(
        "\n{:<14}{:>10}{:>10}{:>8}",
        "class", "precision", "recall", "n"
    );
    for (c, class) in classes.iter().enumerate() {
        let tp = matrix[c][c] as f64;
        let pred = matrix[c].iter().sum::<usize>() as f64;
        let actual = matrix.iter().map(|row| row[c]).sum::<usize>() as f64;
        let p = if pred > 0.0 { tp / pred } else { f64::NAN };
        let r = if actual > 0.0 { tp / actual } else { f64::NAN };
        println!("{:<14}{p:>10.3}{r:>10.3}{actual:>8}", class.as_str());
    }

    // AC-8 bar: precision(supersession) ≥ 0.90 and ≤10% of coexist pairs
    // mis-called contradiction/supersession.
    let s = idx(Verdict::Supersession);
    let c = idx(Verdict::Coexist);
    let k = idx(Verdict::Contradiction);
    let sup_pred: usize = matrix[s].iter().sum();
    let sup_precision = if sup_pred > 0 {
        matrix[s][s] as f64 / sup_pred as f64
    } else {
        f64::NAN
    };
    let coexist_n: usize = matrix.iter().map(|row| row[c]).sum();
    let coexist_escalated = matrix[s][c] + matrix[k][c];
    let coexist_escalation_rate = if coexist_n > 0 {
        coexist_escalated as f64 / coexist_n as f64
    } else {
        f64::NAN
    };
    let survivor_acc = if surv_total > 0 {
        surv_correct as f64 / surv_total as f64
    } else {
        f64::NAN
    };
    println!(
        "\nbar: precision(supersession)={sup_precision:.3} (≥0.90)  coexist→conflict={coexist_escalation_rate:.3} (≤0.10)  survivor_acc(on true supersessions)={survivor_acc:.3}"
    );
    println!(
        "tokens: input={in_tok} output={out_tok} per_pair_in={:.0} per_pair_out={:.1}",
        in_tok as f64 / results.len().max(1) as f64,
        out_tok as f64 / results.len().max(1) as f64
    );
    // Per-source breakdown helps spot label-source bias.
    let mut by_source: HashMap<&str, (usize, usize)> = HashMap::new();
    for (p, j) in &results {
        let e = by_source.entry(p.source.as_str()).or_default();
        e.1 += 1;
        if j.as_ref().is_some_and(|j| j.verdict == p.label) {
            e.0 += 1;
        }
    }
    for (src, (ok, n)) in by_source {
        println!("source {src}: {ok}/{n} exact-class agreement");
    }
    assert_eq!(
        skipped, 0,
        "every fixture pair must resolve to two live memories"
    );
}
