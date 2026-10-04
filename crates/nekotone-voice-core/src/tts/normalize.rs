//! English text normalisation before G2P (owner: the G2P package).
//!
//! Turns written text into speakable words: numbers (cardinals, decimals,
//! negatives, thousands separators), ordinals (1st, 22nd), years (1999 →
//! nineteen ninety-nine), currency ($5.20, £3, €1m), percentages, times
//! (3:45 pm), dates (12/03/2024 and "March 3rd"), ranges (5-10), units
//! (km, kg, °C where common), phone-like digit strings (read digit by digit),
//! abbreviations (Mr., Dr., St., etc., e.g., i.e., vs.), symbols (& + @ % #),
//! URLs and e-mail addresses ("dot", "at"). Quotes, dashes and ellipses are
//! unified to the characters Kokoro knows (`"`, `—`, `…`); other sentence
//! punctuation is kept so the model pauses at it. Acronyms are left as
//! written (the G2P spells unknown ones).
//!
//! Number words are written with spaces, not hyphens ("twenty one"), the
//! way misaki feeds them to its lexicon; "a.m."/"p.m." become the letters
//! "A M"/"P M" so the G2P spells them.

/// Normalise `text`; `british` picks "one hundred and five" style and
/// day/month order for ambiguous dates.
pub fn normalize(text: &str, british: bool) -> String {
    let cleaned = pre_clean(text);
    let chunks: Vec<&str> = cleaned.split_whitespace().collect();
    let mut out: Vec<String> = Vec::with_capacity(chunks.len());
    let mut i = 0;
    while i < chunks.len() {
        i += expand(&chunks, i, british, &mut out);
    }
    let mut s = String::with_capacity(cleaned.len() + 16);
    for piece in out {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        if !s.is_empty() {
            s.push(' ');
        }
        s.push_str(piece);
    }
    s
}

// ───────────────────────────── numbers ─────────────────────────────

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const SCALES: [&str; 7] = ["", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion"];

fn under_100(n: u32) -> String {
    if n < 20 {
        ONES[n as usize].to_string()
    } else if n % 10 == 0 {
        TENS[(n / 10) as usize].to_string()
    } else {
        format!("{} {}", TENS[(n / 10) as usize], ONES[(n % 10) as usize])
    }
}

fn under_1000(n: u32, british: bool) -> String {
    let (h, r) = (n / 100, n % 100);
    match (h, r) {
        (0, r) => under_100(r),
        (h, 0) => format!("{} hundred", ONES[h as usize]),
        (h, r) => format!("{} hundred {}{}", ONES[h as usize], if british { "and " } else { "" }, under_100(r)),
    }
}

/// Cardinal number in words ("one thousand two hundred thirty four";
/// British: "… two hundred and thirty four", "one thousand and five").
pub fn cardinal(n: u128, british: bool) -> String {
    if n == 0 {
        return "zero".into();
    }
    let mut groups = Vec::new();
    let mut m = n;
    while m > 0 {
        groups.push((m % 1000) as u32);
        m /= 1000;
    }
    if groups.len() > SCALES.len() {
        return digits(&n.to_string());
    }
    let mut parts: Vec<String> = Vec::new();
    for (k, &g) in groups.iter().enumerate().rev() {
        if g == 0 {
            continue;
        }
        let mut w = under_1000(g, british);
        if k == 0 && british && groups.len() > 1 && g < 100 {
            w = format!("and {w}");
        }
        if k > 0 {
            w = format!("{w} {}", SCALES[k]);
        }
        parts.push(w);
    }
    parts.join(" ")
}

/// Ordinal in words ("twenty first").
pub fn ordinal(n: u128, british: bool) -> String {
    let c = cardinal(n, british);
    let (head, last) = match c.rfind(' ') {
        Some(i) => (&c[..=i], &c[i + 1..]),
        None => ("", c.as_str()),
    };
    let o = match last {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        w if w.ends_with('y') => format!("{}ieth", &w[..w.len() - 1]),
        w => format!("{w}th"),
    };
    format!("{head}{o}")
}

/// A year: 1999 → nineteen ninety nine, 1905 → nineteen oh five, 2005 →
/// two thousand (and) five, 2024 → twenty twenty four.
pub fn year(n: u32, british: bool) -> String {
    if (2000..=2009).contains(&n) {
        return if n == 2000 { "two thousand".into() } else { format!("two thousand {}{}", if british { "and " } else { "" }, ONES[(n - 2000) as usize]) };
    }
    if !(1000..=9999).contains(&n) || n % 1000 == 0 {
        return cardinal(n as u128, british);
    }
    let (hi, lo) = (n / 100, n % 100);
    if lo == 0 {
        format!("{} hundred", under_100(hi))
    } else if lo < 10 {
        format!("{} oh {}", under_100(hi), ONES[lo as usize])
    } else {
        format!("{} {}", under_100(hi), under_100(lo))
    }
}

/// Digits one by one.
pub fn digits(s: &str) -> String {
    s.chars().filter_map(|c| c.to_digit(10)).map(|d| ONES[d as usize]).collect::<Vec<_>>().join(" ")
}

fn plural_of_number(words: &str) -> String {
    match words.rfind(' ') {
        Some(i) => format!("{} {}", &words[..i], plural_word(&words[i + 1..])),
        None => plural_word(words),
    }
}

fn plural_word(w: &str) -> String {
    if let Some(s) = w.strip_suffix('y') {
        format!("{s}ies")
    } else if w == "six" {
        "sixes".into()
    } else {
        format!("{w}s")
    }
}

/// "1,234" / "1234" → 1234 (commas only as proper thousands separators).
fn parse_int(s: &str) -> Option<u128> {
    if s.is_empty() || s.len() > 36 {
        return None;
    }
    if s.contains(',') {
        let groups: Vec<&str> = s.split(',').collect();
        if groups[0].is_empty() || groups[0].len() > 3 || groups[1..].iter().any(|g| g.len() != 3) {
            return None;
        }
    }
    let d: String = s.chars().filter(|c| *c != ',').collect();
    if d.is_empty() || !d.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    d.parse().ok()
}

fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

/// A plain number: "12", "1,234", "3.14", ".5", "-7", "007".
fn number_words(s: &str, british: bool, year_ok: bool) -> Option<String> {
    let (neg, body) = match s.strip_prefix('-').or_else(|| s.strip_prefix('−')) {
        Some(b) => (true, b),
        None => (false, s),
    };
    if body.is_empty() {
        return None;
    }
    let w = if let Some((a, b)) = body.split_once('.') {
        if !is_digits(b) || (!a.is_empty() && parse_int(a).is_none()) {
            return None;
        }
        let int = if a.is_empty() { String::new() } else { format!("{} ", cardinal(parse_int(a)?, british)) };
        format!("{int}point {}", digits(b))
    } else {
        let digits_only: String = body.chars().filter(|c| *c != ',').collect();
        if !is_digits(&digits_only) {
            return None;
        }
        if (digits_only.len() > 1 && digits_only.starts_with('0')) || (!body.contains(',') && digits_only.len() > 15) {
            digits(&digits_only)
        } else {
            let n = parse_int(body)?;
            if year_ok && !neg && !body.contains(',') && digits_only.len() == 4 && (1100..=2099).contains(&n) {
                year(n as u32, british)
            } else {
                cardinal(n, british)
            }
        }
    };
    Some(if neg { format!("minus {w}") } else { w })
}

// ───────────────────────────── tables ─────────────────────────────

const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

fn month_index(w: &str) -> Option<usize> {
    let l = w.trim_end_matches('.').to_ascii_lowercase();
    if l.len() < 3 {
        return None;
    }
    MONTHS.iter().position(|m| {
        let ml = m.to_ascii_lowercase();
        ml == l || (l.len() >= 3 && ml.starts_with(&l) && (l.len() == 3 || l == "sept"))
    })
}

/// Units after a number: (spellings, singular, plural).
const UNITS: &[(&[&str], &str, &str)] = &[
    (&["km", "kms"], "kilometer", "kilometers"),
    (&["cm"], "centimeter", "centimeters"),
    (&["mm"], "millimeter", "millimeters"),
    (&["kg", "kgs"], "kilogram", "kilograms"),
    (&["mg"], "milligram", "milligrams"),
    (&["lb", "lbs"], "pound", "pounds"),
    (&["oz"], "ounce", "ounces"),
    (&["mph"], "mile per hour", "miles per hour"),
    (&["kph", "km/h", "kmh"], "kilometer per hour", "kilometers per hour"),
    (&["ms"], "millisecond", "milliseconds"),
    (&["sec", "secs"], "second", "seconds"),
    (&["min", "mins"], "minute", "minutes"),
    (&["hr", "hrs"], "hour", "hours"),
    (&["ft"], "foot", "feet"),
    (&["mi"], "mile", "miles"),
    (&["kb", "KB"], "kilobyte", "kilobytes"),
    (&["mb", "MB"], "megabyte", "megabytes"),
    (&["gb", "GB"], "gigabyte", "gigabytes"),
    (&["tb", "TB"], "terabyte", "terabytes"),
    (&["hz", "Hz"], "hertz", "hertz"),
    (&["khz", "kHz"], "kilohertz", "kilohertz"),
    (&["mhz", "MHz"], "megahertz", "megahertz"),
    (&["ghz", "GHz"], "gigahertz", "gigahertz"),
    (&["kw", "kW"], "kilowatt", "kilowatts"),
    (&["fps", "FPS"], "frame per second", "frames per second"),
    (&["°c", "°C", "ºC"], "degree Celsius", "degrees Celsius"),
    (&["°f", "°F", "ºF"], "degree Fahrenheit", "degrees Fahrenheit"),
    (&["°", "º"], "degree", "degrees"),
    (&["usd", "USD"], "dollar", "dollars"),
    (&["eur", "EUR"], "euro", "euros"),
    (&["gbp", "GBP"], "pound", "pounds"),
];

fn unit(s: &str) -> Option<(&'static str, &'static str)> {
    UNITS.iter().find(|(names, _, _)| names.iter().any(|n| *n == s || (n.eq_ignore_ascii_case(s) && n.len() > 2 && !n.starts_with('°')))).map(|(_, a, b)| (*a, *b))
}

fn multiplier(s: &str) -> Option<&'static str> {
    Some(match s {
        "k" | "K" | "thousand" => "thousand",
        "m" | "M" | "mn" | "million" | "mil" => "million",
        "b" | "B" | "bn" | "billion" => "billion",
        "t" | "tn" | "trillion" => "trillion",
        _ => return None,
    })
}

/// Currency symbol → (singular, plural, sub-unit singular, sub-unit plural).
fn currency(c: char) -> Option<(&'static str, &'static str, &'static str, &'static str)> {
    Some(match c {
        '$' => ("dollar", "dollars", "cent", "cents"),
        '£' => ("pound", "pounds", "penny", "pence"),
        '€' => ("euro", "euros", "cent", "cents"),
        '¥' => ("yen", "yen", "sen", "sen"),
        '₹' => ("rupee", "rupees", "paisa", "paise"),
        _ => return None,
    })
}

// ───────────────────────────── cleaning ─────────────────────────────

fn pre_clean(text: &str) -> String {
    let mut s = String::with_capacity(text.len() + 8);
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let last_visible = |s: &String| s.trim_end().chars().last();
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        let prev = if i > 0 { Some(chars[i - 1]) } else { None };
        match c {
            '‘' | '’' | 'ʼ' | '`' | '´' => s.push('\''),
            '«' | '»' | '„' | '‟' => s.push('"'),
            '\n' | '\r' => {
                // a line break after words with no punctuation reads as a sentence end
                if last_visible(&s).map(|c| c.is_alphanumeric()).unwrap_or(false) && chars[i + 1..].iter().any(|c| !c.is_whitespace()) {
                    s.push('.');
                }
                s.push(' ');
            }
            '.' if next == Some('.') && chars.get(i + 2) == Some(&'.') => {
                s.push('…');
                i += 3;
                while chars.get(i) == Some(&'.') {
                    i += 1;
                }
                continue;
            }
            '-' if next == Some('-') => {
                s.push_str(" — ");
                i += 2;
                continue;
            }
            '–' | '‒' | '―' => {
                // between digits it is a range; otherwise a pause
                if prev.map(|p| p.is_ascii_digit()).unwrap_or(false) && next.map(|n| n.is_ascii_digit()).unwrap_or(false) {
                    s.push('-');
                } else {
                    s.push_str(" — ");
                }
            }
            '—' => s.push_str(" — "),
            '-' if prev.map(char::is_whitespace).unwrap_or(true) && next.map(char::is_whitespace).unwrap_or(true) => s.push_str(" — "),
            '\u{00a0}' | '\t' | '\u{2009}' | '\u{202f}' => s.push(' '),
            '−' => s.push('-'),
            '×' => s.push('x'),
            c if c.is_alphanumeric() || c.is_whitespace() => s.push(c),
            c if ".,;:!?'\"“”()[]{}-—…%&+@#$€£¥₹/=*~°º_<>|\\^".contains(c) => s.push(c),
            _ => {
                // emoji and other symbols: a space so neighbours stay apart
                s.push(' ');
            }
        }
        i += 1;
    }
    s
}

const LEAD: &str = "\"'“([{¿¡";
const TRAIL: &str = ".,;:!?\"'”)]}…";

fn split_punct(chunk: &str) -> (&str, &str, &str) {
    let mut start = chunk.char_indices().find(|(_, c)| !LEAD.contains(*c)).map(|(i, _)| i).unwrap_or(chunk.len());
    // keep the apostrophe of "'90s"
    if start > 0 && chunk[..start].ends_with('\'') && chunk[start..].starts_with(|c: char| c.is_ascii_digit()) {
        start -= 1;
    }
    let rest = &chunk[start..];
    let mut end = rest.len();
    for (i, c) in rest.char_indices().rev() {
        if TRAIL.contains(c) {
            // keep the apostrophe of a plural possessive ("dogs'")
            if c == '\'' && i > 0 && rest[..i].ends_with(['s', 'S']) && i + 1 == end && !rest[..i].contains('\'') {
                break;
            }
            end = i;
        } else {
            break;
        }
    }
    (&chunk[..start], &rest[..end], &rest[end..])
}

// ───────────────────────────── abbreviations ─────────────────────────────

/// Titles before a name: the period is never a sentence end.
fn title(core: &str) -> Option<&'static str> {
    Some(match core {
        "Mr" => "Mister",
        "Mrs" => "Missus",
        "Ms" => "Miz",
        "Dr" => "Doctor",
        "Prof" => "Professor",
        "Rev" => "Reverend",
        "Gen" => "General",
        "Capt" => "Captain",
        "Lt" => "Lieutenant",
        "Sgt" => "Sergeant",
        "Col" => "Colonel",
        "Gov" => "Governor",
        "Sen" => "Senator",
        "Pres" => "President",
        _ => return None,
    })
}

/// Abbreviations whose period may also end the sentence.
fn abbreviation(core: &str, has_dot: bool) -> Option<&'static str> {
    let with_dot_only = match core {
        "etc" => "et cetera",
        "Jr" | "jr" => "Junior",
        "Sr" | "sr" => "Senior",
        "Inc" => "Incorporated",
        "Ltd" => "Limited",
        "Corp" => "Corporation",
        "Co" => "Company",
        "Ave" => "Avenue",
        "Blvd" => "Boulevard",
        "Rd" => "Road",
        "Dept" => "Department",
        "approx" => "approximately",
        "Approx" => "Approximately",
        "misc" => "miscellaneous",
        "est" => "established",
        _ => "",
    };
    if has_dot && !with_dot_only.is_empty() {
        return Some(with_dot_only);
    }
    Some(match core {
        "e.g" | "eg" if has_dot || core.contains('.') => "for example",
        "E.g" => "For example",
        "i.e" | "ie" if has_dot || core.contains('.') => "that is",
        "I.e" => "That is",
        "vs" | "Vs" | "VS" | "v" => {
            if core == "v" && !has_dot {
                return None;
            }
            "versus"
        }
        "w/" => "with",
        "w/o" => "without",
        "pls" | "plz" => "please",
        "Pls" | "Plz" => "Please",
        "thx" => "thanks",
        "Thx" => "Thanks",
        "a.m" | "A.M" | "am" if has_dot => "A M",
        "p.m" | "P.M" | "pm" if has_dot => "P M",
        _ => return None,
    })
}

// ───────────────────────────── the chunk expander ─────────────────────────────

fn starts_upper(s: &str) -> bool {
    s.chars().find(|c| c.is_alphanumeric()).map(|c| c.is_uppercase()).unwrap_or(false)
}

/// Expand `chunks[i]` (looking at neighbours); returns how many chunks it used.
fn expand(chunks: &[&str], i: usize, br: bool, out: &mut Vec<String>) -> usize {
    let chunk = chunks[i];
    let (lead, core, trail) = split_punct(chunk);
    let next = chunks.get(i + 1).copied();
    let next_core = next.map(|n| split_punct(n).1);
    let push = |out: &mut Vec<String>, body: String, trail: &str| out.push(format!("{lead}{body}{trail}"));
    if core.is_empty() {
        out.push(chunk.to_string());
        return 1;
    }
    let trail_dot = trail.starts_with('.');
    let rest_after_dot = if trail_dot { &trail[1..] } else { trail };
    let sentence_end_after = next.is_none() || next.map(starts_upper).unwrap_or(false);

    // Titles: "Dr. Smith"
    if let Some(t) = title(core) {
        if trail_dot || next.map(starts_upper).unwrap_or(false) {
            push(out, t.into(), rest_after_dot);
            return 1;
        }
    }
    // "St." → Saint before a name, Street otherwise
    if core == "St" && (trail_dot || next.is_some()) {
        if next.map(starts_upper).unwrap_or(false) && !rest_after_dot.contains([',', ';']) {
            push(out, "Saint".into(), rest_after_dot);
        } else {
            let keep = if trail_dot && sentence_end_after { trail } else { rest_after_dot };
            push(out, "Street".into(), keep);
        }
        return 1;
    }
    if (core == "Mt" || core == "Ft") && next.map(starts_upper).unwrap_or(false) {
        push(out, if core == "Mt" { "Mount".into() } else { "Fort".into() }, rest_after_dot);
        return 1;
    }
    // "No. 5" → number five
    if (core == "No" || core == "no" || core == "Nos" || core == "nos") && trail_dot && next_core.map(|n| n.starts_with(|c: char| c.is_ascii_digit())).unwrap_or(false) {
        push(out, if core.starts_with('N') { "Number".into() } else { "number".into() }, rest_after_dot);
        return 1;
    }
    // months with a period ("Jan.")
    if trail_dot && core.len() <= 4 && core.chars().next().map(char::is_uppercase).unwrap_or(false) {
        if let Some(m) = month_index(core) {
            if core.len() < MONTHS[m].len() {
                let keep = if sentence_end_after && !next_core.map(|n| n.starts_with(|c: char| c.is_ascii_digit())).unwrap_or(false) { trail } else { rest_after_dot };
                push(out, MONTHS[m].into(), keep);
                return 1;
            }
        }
    }
    if let Some(a) = abbreviation(core, trail_dot) {
        let keep = if trail_dot && sentence_end_after && !matches!(a, "for example" | "that is" | "versus") {
            trail
        } else {
            rest_after_dot
        };
        push(out, a.into(), keep);
        return 1;
    }

    // URLs and e-mail
    if let Some(u) = url_words(core) {
        push(out, u, trail);
        return 1;
    }

    // hashtags and mentions
    if let Some(r) = core.strip_prefix('#') {
        if r.starts_with(|c: char| c.is_ascii_digit()) {
            let n = expand_core(r, br, None, next_core, out.last().map(String::as_str)).map(|(w, _)| w).unwrap_or_else(|| r.to_string());
            push(out, format!("number {n}"), trail);
            return 1;
        }
        if r.chars().next().map(char::is_alphabetic).unwrap_or(false) {
            push(out, format!("hashtag {}", split_camel(r)), trail);
            return 1;
        }
    }
    if let Some(r) = core.strip_prefix('@') {
        if r.chars().next().map(char::is_alphanumeric).unwrap_or(false) {
            push(out, format!("at {}", r.replace('_', " ")), trail);
            return 1;
        }
    }

    // currency with a separate multiplier word or code: "$1.5 million", "5 USD"
    if let Some(first) = core.chars().next() {
        if currency(first).is_some() {
            if let Some(m) = next_core.and_then(multiplier).filter(|m| m.len() > 1 && next_core.map(|n| n.len() > 2).unwrap_or(false)) {
                let next_trail = split_punct(next.unwrap_or("")).2;
                if let Some(w) = money(core, br, Some(m)) {
                    push(out, w, next_trail);
                    return 2;
                }
                let _ = m;
            }
        }
    }

    let prev_out = out.last().map(String::as_str);
    if let Some((w, used_next)) = expand_core(core, br, next, next_core, prev_out) {
        if used_next {
            let next_trail = split_punct(next.unwrap_or("")).2;
            push(out, w, next_trail);
            return 2;
        }
        push(out, w, trail);
        return 1;
    }
    push(out, core.to_string(), trail);
    1
}

/// "HelloWorld" → "Hello World" (hashtags).
fn split_camel(s: &str) -> String {
    let mut out = String::new();
    let mut prev: Option<char> = None;
    for c in s.chars() {
        if c == '_' {
            out.push(' ');
        } else {
            if let Some(p) = prev {
                if c.is_uppercase() && p.is_lowercase() {
                    out.push(' ');
                }
            }
            out.push(c);
        }
        prev = Some(c);
    }
    out
}

fn url_words(core: &str) -> Option<String> {
    const TLDS: &[&str] = &["com", "org", "net", "io", "gg", "co", "uk", "de", "tv", "edu", "gov", "app", "dev", "ai", "me", "info", "fr", "eu", "us", "ca"];
    let lower = core.to_ascii_lowercase();
    let (body, is_url) = if let Some(r) = lower.strip_prefix("https://").or_else(|| lower.strip_prefix("http://")) {
        (r.to_string(), true)
    } else if lower.starts_with("www.") {
        (lower.clone(), true)
    } else {
        (lower.clone(), false)
    };
    let email = !is_url && {
        let parts: Vec<&str> = body.split('@').collect();
        parts.len() == 2 && !parts[0].is_empty() && parts[1].contains('.') && parts[1].split('.').all(|p| !p.is_empty())
    };
    let bare_domain = !is_url && !email && {
        let host = body.split('/').next().unwrap_or("");
        let labels: Vec<&str> = host.split('.').collect();
        labels.len() >= 2
            && labels.iter().all(|l| !l.is_empty() && l.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'))
            && TLDS.contains(labels.last().unwrap_or(&""))
            && labels[0].chars().count() >= 2
    };
    if !is_url && !email && !bare_domain {
        return None;
    }
    let body = body.trim_end_matches('/');
    let mut words: Vec<String> = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, words: &mut Vec<String>| {
        if !cur.is_empty() {
            if cur == "www" {
                words.push("W W W".into());
            } else if cur.chars().all(|c| c.is_ascii_digit()) {
                words.push(digits(cur));
            } else {
                words.push(cur.clone());
            }
            cur.clear();
        }
    };
    for c in body.chars() {
        let sep = match c {
            '.' => Some("dot"),
            '/' => Some("slash"),
            '@' => Some("at"),
            '-' => Some("dash"),
            '_' => Some("underscore"),
            ':' => Some("colon"),
            '?' => Some("question mark"),
            '=' => Some("equals"),
            '&' => Some("and"),
            _ => None,
        };
        match sep {
            Some(w) => {
                flush(&mut cur, &mut words);
                words.push(w.into());
            }
            None => cur.push(c),
        }
    }
    flush(&mut cur, &mut words);
    Some(words.join(" "))
}

/// Money: "$5", "$5.20", "$1,000", "£3.50", "€2.5m", "$5k", "5€".
fn money(core: &str, br: bool, mult_word: Option<&str>) -> Option<String> {
    let (neg, core) = match core.strip_prefix('-') {
        Some(c) => (true, c),
        None => (false, core),
    };
    let first = core.chars().next()?;
    let last = core.chars().last()?;
    let (sym, amount) = if currency(first).is_some() {
        (first, &core[first.len_utf8()..])
    } else if currency(last).is_some() {
        (last, &core[..core.len() - last.len_utf8()])
    } else {
        return None;
    };
    let (one, many, sub1, subn) = currency(sym)?;
    // split a trailing multiplier ("2.5m", "5k", "3bn")
    let split = amount.find(|c: char| c.is_alphabetic()).unwrap_or(amount.len());
    let (num, suffix) = amount.split_at(split);
    let mult = if suffix.is_empty() { mult_word } else { Some(multiplier(suffix)?) };
    if num.is_empty() {
        return None;
    }
    let minus = if neg { "minus " } else { "" };
    if let Some(m) = mult {
        let n = number_words(num, br, false)?;
        return Some(format!("{minus}{n} {m} {many}"));
    }
    let (int_s, cents_s) = match num.split_once('.') {
        Some((a, b)) => (a, Some(b)),
        None => (num, None),
    };
    let int = if int_s.is_empty() { 0 } else { parse_int(int_s)? };
    let cents: Option<u128> = match cents_s {
        Some(c) if c.len() == 2 && is_digits(c) => Some(c.parse().ok()?),
        Some(c) if c.len() == 1 && is_digits(c) => Some(c.parse::<u128>().ok()? * 10),
        Some("") => None,
        Some(_) => {
            // odd precision: say it as a decimal
            let n = number_words(num, br, false)?;
            return Some(format!("{minus}{n} {many}"));
        }
        None => None,
    };
    let unit_word = |n: u128, a: &str, b: &str| if n == 1 { a.to_string() } else { b.to_string() };
    let mut s = String::from(minus);
    match cents {
        Some(c) if c > 0 && int > 0 => {
            s.push_str(&format!("{} {} and {} {}", cardinal(int, br), unit_word(int, one, many), cardinal(c, br), unit_word(c, sub1, subn)));
        }
        Some(c) if c > 0 => s.push_str(&format!("{} {}", cardinal(c, br), unit_word(c, sub1, subn))),
        _ => s.push_str(&format!("{} {}", cardinal(int, br), unit_word(int, one, many))),
    }
    Some(s)
}

/// Time "3:45", "15:00", "3:45pm", "3pm", "10:30:15".
fn time_words(core: &str, br: bool, next_core: Option<&str>) -> Option<(String, bool)> {
    let lower = core.to_ascii_lowercase();
    let (body, mut ampm) = if let Some(b) = lower.strip_suffix("am").or_else(|| lower.strip_suffix("a.m")) {
        (b.to_string(), Some("A M"))
    } else if let Some(b) = lower.strip_suffix("pm").or_else(|| lower.strip_suffix("p.m")) {
        (b.to_string(), Some("P M"))
    } else {
        (lower.clone(), None)
    };
    let mut used_next = false;
    if ampm.is_none() {
        if let Some(n) = next_core {
            let nl = n.to_ascii_lowercase();
            if matches!(nl.as_str(), "am" | "a.m" | "a.m." | "a.m.." ) {
                ampm = Some("A M");
                used_next = true;
            } else if matches!(nl.as_str(), "pm" | "p.m" | "p.m.") {
                ampm = Some("P M");
                used_next = true;
            }
        }
    }
    let parts: Vec<&str> = body.split(':').collect();
    if parts.iter().any(|p| !is_digits(p)) || parts[0].len() > 2 {
        return None;
    }
    let h: u32 = parts[0].parse().ok()?;
    if parts.len() == 1 {
        // "3pm"
        let a = ampm?;
        if h == 0 || h > 12 {
            return None;
        }
        return Some((format!("{} {a}", cardinal(h as u128, br)), used_next));
    }
    if parts.len() > 3 || parts[1..].iter().any(|p| p.len() != 2) {
        return None;
    }
    let m: u32 = parts[1].parse().ok()?;
    if h > 24 || m > 59 {
        return None;
    }
    let mut s = cardinal(h as u128, br);
    if m == 0 {
        if ampm.is_none() {
            if h > 12 || h == 0 || parts[0].len() == 2 && parts[0].starts_with('0') {
                s = if h == 0 { "zero hundred".into() } else { format!("{s} hundred") };
            } else {
                s.push_str(" o'clock");
            }
        }
    } else if m < 10 {
        s.push_str(&format!(" oh {}", ONES[m as usize]));
    } else {
        s.push(' ');
        s.push_str(&under_100(m));
    }
    if parts.len() == 3 {
        let sec: u32 = parts[2].parse().ok()?;
        s.push_str(&format!(" and {} {}", cardinal(sec as u128, br), if sec == 1 { "second" } else { "seconds" }));
    }
    if let Some(a) = ampm {
        s.push(' ');
        s.push_str(a);
    }
    Some((s, used_next))
}

fn date_words(day: u32, month: usize, year_v: Option<u32>, br: bool) -> String {
    let m = MONTHS[month];
    let d = ordinal(day as u128, br);
    match (br, year_v) {
        (true, Some(y)) => format!("the {d} of {m}, {}", year(y, br)),
        (true, None) => format!("the {d} of {m}"),
        (false, Some(y)) => format!("{m} {d}, {}", year(y, br)),
        (false, None) => format!("{m} {d}"),
    }
}

/// "2024-03-12", "12/03/2024", "3/12/24", "12.03.2024".
fn date(core: &str, br: bool) -> Option<String> {
    let sep = ['-', '/', '.'].into_iter().find(|s| core.contains(*s))?;
    let parts: Vec<&str> = core.split(sep).collect();
    if parts.len() != 3 || parts.iter().any(|p| !is_digits(p)) {
        return None;
    }
    let n: Vec<u32> = parts.iter().map(|p| p.parse().unwrap_or(0)).collect();
    if parts[0].len() == 4 {
        let (y, m, d) = (n[0], n[1], n[2]);
        if (1..=12).contains(&m) && (1..=31).contains(&d) && sep == '-' || (sep == '/' && (1..=12).contains(&m) && (1..=31).contains(&d)) {
            return Some(date_words(d, m as usize - 1, Some(y), br));
        }
        return None;
    }
    if parts[2].len() != 4 && parts[2].len() != 2 {
        return None;
    }
    if sep == '-' && parts[2].len() == 2 {
        return None; // more likely a phone number or a code
    }
    let y = if parts[2].len() == 2 { if n[2] < 50 { 2000 + n[2] } else { 1900 + n[2] } } else { n[2] };
    let (a, b) = (n[0], n[1]);
    let (d, m) = if a > 12 && b <= 12 {
        (a, b)
    } else if b > 12 && a <= 12 {
        (b, a)
    } else if br {
        (a, b)
    } else {
        (b, a)
    };
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    Some(date_words(d, m as usize - 1, Some(y), br))
}

fn fraction(n: u128, d: u128, br: bool) -> Option<String> {
    if d == 0 || n == 0 || n >= d || d > 16 {
        return None;
    }
    let name = match d {
        2 => {
            if n == 1 {
                "half".to_string()
            } else {
                "halves".to_string()
            }
        }
        4 => {
            if n == 1 {
                "quarter".to_string()
            } else {
                "quarters".to_string()
            }
        }
        _ => {
            let o = ordinal(d, br);
            if n == 1 {
                o
            } else {
                format!("{o}s")
            }
        }
    };
    Some(format!("{} {name}", if n == 1 && d == 2 { "one".to_string() } else { cardinal(n, br) }))
}

fn is_phone_like(core: &str) -> bool {
    let digits_n = core.chars().filter(|c| c.is_ascii_digit()).count();
    let groups = core.split(|c: char| !c.is_ascii_digit()).filter(|g| !g.is_empty()).count();
    digits_n >= 7 && groups >= 2 && core.chars().all(|c| c.is_ascii_digit() || "-+(). ".contains(c))
}

fn phone(core: &str) -> String {
    let mut groups = Vec::new();
    if core.starts_with('+') {
        groups.push("plus".to_string());
    }
    for g in core.split(|c: char| !c.is_ascii_digit()).filter(|g| !g.is_empty()) {
        groups.push(digits(g));
    }
    // "plus one, five five five, …"
    let mut s = String::new();
    for (i, g) in groups.iter().enumerate() {
        if i > 0 {
            s.push_str(if groups[i - 1] == "plus" { " " } else { ", " });
        }
        s.push_str(g);
    }
    s
}

/// Everything that starts with or contains digits (and currency).
/// Returns the words and whether the next chunk was used (a unit, am/pm, a month).
fn expand_core(core: &str, br: bool, next: Option<&str>, next_core: Option<&str>, prev_out: Option<&str>) -> Option<(String, bool)> {
    let _ = next;
    if !core.chars().any(|c| c.is_ascii_digit()) {
        return symbols_in_word(core);
    }
    // currency
    if core.chars().next().map(|c| currency(c).is_some()).unwrap_or(false)
        || core.strip_prefix('-').and_then(|c| c.chars().next()).map(|c| currency(c).is_some()).unwrap_or(false)
        || core.chars().last().map(|c| currency(c).is_some()).unwrap_or(false)
    {
        if let Some(m) = money(core, br, None) {
            return Some((m, false));
        }
    }
    // number + currency code or unit as the next word: "5 USD", "10 km"
    let prev_month = prev_out.and_then(|p| p.split_whitespace().last()).and_then(month_index);
    // time
    if core.contains(':') || core.to_ascii_lowercase().ends_with("am") || core.to_ascii_lowercase().ends_with("pm") {
        if let Some((t, used)) = time_words(core, br, next_core) {
            return Some((t, used));
        }
    }
    if let Some(n) = next_core {
        let nl = n.to_ascii_lowercase();
        if is_digits(core) && core.len() <= 2 && matches!(nl.as_str(), "am" | "pm" | "a.m" | "p.m" | "a.m." | "p.m.") {
            if let Some(t) = time_words(&format!("{core}{nl}"), br, None) {
                return Some((t.0, true));
            }
        }
    }
    // dates
    if let Some(d) = date(core, br) {
        return Some((d, false));
    }
    // percent
    if let Some(p) = core.strip_suffix('%') {
        if let Some(w) = number_words(p, br, false) {
            return Some((format!("{w} percent"), false));
        }
        if let Some((a, b)) = p.split_once('-') {
            if let (Some(x), Some(y)) = (number_words(a, br, false), number_words(b.trim_end_matches('%'), br, false)) {
                return Some((format!("{x} to {y} percent"), false));
            }
        }
    }
    // ordinals "1st", "22nd"
    let lower = core.to_ascii_lowercase();
    for suf in ["st", "nd", "rd", "th"] {
        if let Some(n) = lower.strip_suffix(suf) {
            if let Some(v) = parse_int(n) {
                let o = ordinal(v, br);
                // "3rd March" → the third of March
                if let Some(m) = next_core.and_then(month_index) {
                    if next_core.map(|x| x.len() >= 3).unwrap_or(false) {
                        return Some((format!("the {o} of {}", MONTHS[m]), true));
                    }
                }
                return Some((o, false));
            }
        }
    }
    // decades "1990s", "90s", "'90s", "80's"
    let dec = lower.trim_start_matches('\'');
    if let Some(n) = dec.strip_suffix("'s").or_else(|| dec.strip_suffix('s')) {
        if is_digits(n) && (n.len() == 4 || n.len() == 2) && n.ends_with('0') {
            let v: u32 = n.parse().ok()?;
            let w = if n.len() == 4 { year(v, br) } else { under_100(v) };
            return Some((plural_of_number(&w), false));
        }
    }
    // phone numbers
    if is_phone_like(core) && date(core, br).is_none() {
        let parts: Vec<&str> = core.split('-').collect();
        let range_of_years = parts.len() == 2 && parts.iter().all(|p| p.len() == 4 && is_digits(p));
        if !range_of_years {
            return Some((phone(core), false));
        }
    }
    // ranges "5-10", "1990-1995"
    if let Some((a, b)) = core.split_once('-') {
        if !a.is_empty() && !b.is_empty() && a.chars().all(|c| c.is_ascii_digit() || c == '.' || c == ',') && b.chars().all(|c| c.is_ascii_digit() || c == '.' || c == ',') {
            let (x, y) = (number_words(a, br, true)?, number_words(b, br, true)?);
            return Some((format!("{x} to {y}"), false));
        }
    }
    // fractions "1/2", "3/4"; "24/7"
    if let Some((a, b)) = core.split_once('/') {
        if is_digits(a) && is_digits(b) {
            if core == "24/7" {
                return Some(("twenty four seven".into(), false));
            }
            let (x, y) = (parse_int(a)?, parse_int(b)?);
            if let Some(f) = fraction(x, y, br) {
                return Some((f, false));
            }
            if (1..=31).contains(&x) && (1..=12).contains(&y) || (1..=12).contains(&x) && (1..=31).contains(&y) {
                let (d, m) = if br || x > 12 { (x, y) } else { (y, x) };
                if (1..=12).contains(&m) {
                    return Some((date_words(d as u32, m as usize - 1, None, br), false));
                }
            }
            return Some((format!("{} slash {}", cardinal(x, br), cardinal(y, br)), false));
        }
    }
    // multiplication "4x4", "1920x1080"; "2x" (times); "x2"
    if let Some((a, b)) = lower.split_once('x') {
        if is_digits(a) && is_digits(b) {
            return Some((format!("{} by {}", cardinal(parse_int(a)?, br), cardinal(parse_int(b)?, br)), false));
        }
        if is_digits(a) && b.is_empty() {
            return Some((format!("{} times", cardinal(parse_int(a)?, br)), false));
        }
        if a.is_empty() && is_digits(b) {
            return Some((format!("times {}", cardinal(parse_int(b)?, br)), false));
        }
    }
    // plain number, possibly with a unit or multiplier next or attached
    if let Some(w) = number_words(core, br, true) {
        let singular = core.trim_start_matches('-') == "1";
        if let Some(n) = next_core {
            if let Some((one, many)) = unit(n) {
                return Some((format!("{} {}", number_words(core, br, false)?, if singular { one } else { many }), true));
            }
            // "3 March" → the third of March
            if let Some(m) = month_index(n) {
                if n.len() >= 3 && is_digits(core) && core.len() <= 2 {
                    let v = parse_int(core)?;
                    if (1..=31).contains(&v) {
                        return Some((format!("the {} of {}", ordinal(v, br), MONTHS[m]), true));
                    }
                }
            }
        }
        // "March 3" → March third; "March 3, 2024"
        if prev_month.is_some() && is_digits(core) && core.len() <= 2 {
            let v = parse_int(core)?;
            if (1..=31).contains(&v) {
                return Some((ordinal(v, br), false));
            }
        }
        return Some((w, false));
    }
    // number with an attached unit / multiplier / letters: "5km", "10k", "1080p", "mp3", "COVID19"
    let split = core.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == ',' || c == '-')).unwrap_or(core.len());
    let (num, tail) = core.split_at(split);
    if !num.is_empty() && !tail.is_empty() && num.chars().any(|c| c.is_ascii_digit()) {
        if let Some((one, many)) = unit(tail) {
            let w = number_words(num, br, false)?;
            return Some((format!("{w} {}", if num == "1" { one } else { many }), false));
        }
        if let Some(m) = multiplier(tail).filter(|_| tail != "M" && tail != "B" && tail != "T") {
            let w = number_words(num, br, false)?;
            return Some((format!("{w} {m}"), false));
        }
    }
    // mixed letters and digits: split into runs
    let mut words: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_digit: Option<bool> = None;
    let flush = |cur: &mut String, d: Option<bool>, words: &mut Vec<String>| {
        if cur.is_empty() {
            return;
        }
        if d == Some(true) {
            let w = if cur.len() == 4 && !cur.starts_with('0') {
                year(cur.parse().unwrap_or(0), br)
            } else if cur.len() < 4 {
                number_words(cur, br, false).unwrap_or_else(|| digits(cur))
            } else {
                digits(cur)
            };
            words.push(w);
        } else {
            words.push(cur.clone());
        }
        cur.clear();
    };
    for c in core.chars() {
        if c.is_ascii_digit() || c.is_alphabetic() {
            let d = c.is_ascii_digit();
            if cur_digit != Some(d) {
                flush(&mut cur, cur_digit, &mut words);
                cur_digit = Some(d);
            }
            cur.push(c);
        } else {
            flush(&mut cur, cur_digit, &mut words);
            cur_digit = None;
            match c {
                '+' => words.push("plus".into()),
                '&' => words.push("and".into()),
                '%' => words.push("percent".into()),
                '=' => words.push("equals".into()),
                '@' => words.push("at".into()),
                '/' => words.push("slash".into()),
                '.' => words.push("point".into()),
                ',' => {
                    if let Some(l) = words.last_mut() {
                        l.push(',');
                    }
                }
                _ => {}
            }
        }
    }
    flush(&mut cur, cur_digit, &mut words);
    if words.is_empty() {
        None
    } else {
        Some((words.join(" "), false))
    }
}

/// Symbols inside or around words without digits: "rock&roll", "and/or",
/// lone "&", "+", "=", "~", "*".
fn symbols_in_word(core: &str) -> Option<(String, bool)> {
    match core {
        "&" => return Some(("and".into(), false)),
        "+" => return Some(("plus".into(), false)),
        "=" => return Some(("equals".into(), false)),
        "@" => return Some(("at".into(), false)),
        "%" => return Some(("percent".into(), false)),
        "#" => return Some(("number".into(), false)),
        "/" | "\\" | "*" | "~" | "|" | "_" | "^" | "<" | ">" | "°" => return Some((String::new(), false)),
        _ => {}
    }
    if !core.chars().any(|c| "&+=/*~_|\\<>^$€£#".contains(c)) {
        return None;
    }
    let mut s = String::new();
    for c in core.chars() {
        match c {
            '&' => s.push_str(" and "),
            '+' => s.push_str(" plus "),
            '=' => s.push_str(" equals "),
            '/' | '\\' | '_' | '|' => s.push(' '),
            '~' | '*' | '<' | '>' | '^' | '#' | '$' | '€' | '£' => {}
            c => s.push(c),
        }
    }
    Some((s.split_whitespace().collect::<Vec<_>>().join(" "), false))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn us(s: &str) -> String {
        normalize(s, false)
    }
    fn gb(s: &str) -> String {
        normalize(s, true)
    }

    #[test]
    fn cardinals_and_ordinals() {
        assert_eq!(cardinal(0, false), "zero");
        assert_eq!(cardinal(21, false), "twenty one");
        assert_eq!(cardinal(105, false), "one hundred five");
        assert_eq!(cardinal(105, true), "one hundred and five");
        assert_eq!(cardinal(1005, true), "one thousand and five");
        assert_eq!(cardinal(1_234_567, false), "one million two hundred thirty four thousand five hundred sixty seven");
        assert_eq!(cardinal(3_000_000_000_000, false), "three trillion");
        assert_eq!(ordinal(1, false), "first");
        assert_eq!(ordinal(22, false), "twenty second");
        assert_eq!(ordinal(20, false), "twentieth");
        assert_eq!(ordinal(112, false), "one hundred twelfth");
        assert_eq!(us("I came 1st and she was 22nd."), "I came first and she was twenty second.");
    }

    #[test]
    fn plain_numbers() {
        assert_eq!(us("I have 3 cats."), "I have three cats.");
        assert_eq!(us("1,234 people"), "one thousand two hundred thirty four people");
        assert_eq!(us("pi is 3.14"), "pi is three point one four");
        assert_eq!(us("-7 degrees"), "minus seven degrees");
        assert_eq!(us(".5"), "point five");
        assert_eq!(us("agent 007"), "agent zero zero seven");
        assert_eq!(us("1234567890123456789"), digits("1234567890123456789"));
        assert_eq!(us("1,2,3"), "one, two, three");
    }

    #[test]
    fn years_and_decades() {
        assert_eq!(us("in 1999"), "in nineteen ninety nine");
        assert_eq!(us("in 1905"), "in nineteen oh five");
        assert_eq!(us("in 1900"), "in nineteen hundred");
        assert_eq!(us("in 2005"), "in two thousand five");
        assert_eq!(gb("in 2005"), "in two thousand and five");
        assert_eq!(us("in 2024"), "in twenty twenty four");
        assert_eq!(us("the 1990s"), "the nineteen nineties");
        assert_eq!(us("the '80s"), "the eighties");
        assert_eq!(us("2500 points"), "two thousand five hundred points");
    }

    #[test]
    fn currency() {
        assert_eq!(us("$5"), "five dollars");
        assert_eq!(us("$1"), "one dollar");
        assert_eq!(us("It costs $5.20."), "It costs five dollars and twenty cents.");
        assert_eq!(us("$0.50"), "fifty cents");
        assert_eq!(us("$5.00"), "five dollars");
        assert_eq!(us("£3"), "three pounds");
        assert_eq!(us("£2.50"), "two pounds and fifty pence");
        assert_eq!(us("€1m"), "one million euros");
        assert_eq!(us("$2.5bn"), "two point five billion dollars");
        assert_eq!(us("$5k"), "five thousand dollars");
        assert_eq!(us("$1.5 million"), "one point five million dollars");
        assert_eq!(us("$1,000,000"), "one million dollars");
        assert_eq!(us("5€"), "five euros");
        assert_eq!(us("5 USD"), "five dollars");
    }

    #[test]
    fn percent_units_ranges() {
        assert_eq!(us("50%"), "fifty percent");
        assert_eq!(us("3.5%"), "three point five percent");
        assert_eq!(us("10-20%"), "ten to twenty percent");
        assert_eq!(us("5-10 minutes"), "five to ten minutes");
        assert_eq!(us("5–10"), "five to ten");
        assert_eq!(us("1990-1995"), "nineteen ninety to nineteen ninety five");
        assert_eq!(us("5km"), "five kilometers");
        assert_eq!(us("1 km"), "one kilometer");
        assert_eq!(us("60 fps"), "sixty frames per second");
        assert_eq!(us("16GB"), "sixteen gigabytes");
        assert_eq!(us("30°C"), "thirty degrees Celsius");
        assert_eq!(us("100 mph"), "one hundred miles per hour");
        assert_eq!(us("2x speed"), "two times speed");
        assert_eq!(us("4x4"), "four by four");
        assert_eq!(us("1/2"), "one half");
        assert_eq!(us("3/4"), "three quarters");
        assert_eq!(us("2/3"), "two thirds");
        assert_eq!(us("24/7"), "twenty four seven");
        assert_eq!(us("10k"), "ten thousand");
    }

    #[test]
    fn times() {
        assert_eq!(us("at 3:45"), "at three forty five");
        assert_eq!(us("at 3:05"), "at three oh five");
        assert_eq!(us("at 3:00"), "at three o'clock");
        assert_eq!(us("at 15:00"), "at fifteen hundred");
        assert_eq!(us("at 15:30"), "at fifteen thirty");
        assert_eq!(us("at 3:45pm"), "at three forty five P M");
        assert_eq!(us("at 3:45 p.m."), "at three forty five P M.");
        assert_eq!(us("at 3pm"), "at three P M");
        assert_eq!(us("at 7 am tomorrow"), "at seven A M tomorrow");
        assert_eq!(us("Meet at 9:30 AM."), "Meet at nine thirty A M.");
    }

    #[test]
    fn dates() {
        assert_eq!(us("2024-03-12"), "March twelfth, twenty twenty four");
        assert_eq!(gb("2024-03-12"), "the twelfth of March, twenty twenty four");
        assert_eq!(us("12/03/2024"), "December third, twenty twenty four");
        assert_eq!(gb("12/03/2024"), "the twelfth of March, twenty twenty four");
        assert_eq!(us("25/12/2024"), "December twenty fifth, twenty twenty four");
        assert_eq!(us("on March 3rd"), "on March third");
        assert_eq!(us("on March 3, 2024"), "on March third, twenty twenty four");
        assert_eq!(us("on 3 March"), "on the third of March");
        assert_eq!(us("the 21st of March"), "the twenty first of March");
        assert_eq!(us("Jan. 5"), "January fifth");
    }

    #[test]
    fn phone_numbers() {
        assert_eq!(us("call 555-123-4567"), "call five five five, one two three, four five six seven");
        assert_eq!(us("+1 555-123-4567"), "plus one five five five, one two three, four five six seven");
    }

    #[test]
    fn abbreviations() {
        assert_eq!(us("Mr. Smith and Dr. Jones"), "Mister Smith and Doctor Jones");
        assert_eq!(us("Mrs Brown"), "Missus Brown");
        assert_eq!(us("St. Louis"), "Saint Louis");
        assert_eq!(us("on Baker St."), "on Baker Street.");
        assert_eq!(us("on Baker St. today"), "on Baker Street today");
        assert_eq!(us("apples, pears, etc. and more"), "apples, pears, et cetera and more");
        assert_eq!(us("apples, pears, etc."), "apples, pears, et cetera.");
        assert_eq!(us("fruit, e.g. apples"), "fruit, for example apples");
        assert_eq!(us("that is, i.e. this"), "that is, that is this");
        assert_eq!(us("cats vs. dogs"), "cats versus dogs");
        assert_eq!(us("cats vs dogs"), "cats versus dogs");
        assert_eq!(us("No. 5"), "Number five");
        assert_eq!(us("say no. Please"), "say no. Please");
        assert_eq!(us("coffee w/ milk"), "coffee with milk");
        assert_eq!(us("pls help"), "please help");
    }

    #[test]
    fn symbols_urls_emails() {
        assert_eq!(us("rock & roll"), "rock and roll");
        assert_eq!(us("R&D"), "R and D");
        assert_eq!(us("and/or"), "and or");
        assert_eq!(us("2 + 2 = 4"), "two plus two equals four");
        assert_eq!(us("#1 fan"), "number one fan");
        assert_eq!(us("#blessed"), "hashtag blessed");
        assert_eq!(us("@nekotone"), "at nekotone");
        assert_eq!(us("visit https://www.example.com/docs"), "visit W W W dot example dot com slash docs");
        assert_eq!(us("see example.com."), "see example dot com.");
        assert_eq!(us("mail john.doe@gmail.com"), "mail john dot doe at gmail dot com");
    }

    #[test]
    fn punctuation_and_cleanup() {
        assert_eq!(us("Wait... what?"), "Wait… what?");
        assert_eq!(us("yes -- no"), "yes — no");
        assert_eq!(us("yes - no"), "yes — no");
        assert_eq!(us("yes – no"), "yes — no");
        assert_eq!(us("it’s “fine”"), "it's “fine”");
        assert_eq!(us("hello 😀 world"), "hello world");
        assert_eq!(us("line one\nline two"), "line one. line two");
        assert_eq!(us("  lots   of\tspace  "), "lots of space");
        assert_eq!(us("(5)"), "(five)");
        assert_eq!(us("\"10\""), "\"ten\"");
        assert_eq!(us(""), "");
    }

    #[test]
    fn mixed_letters_and_digits() {
        assert_eq!(us("1080p"), "ten eighty p");
        assert_eq!(us("mp3"), "mp three");
        assert_eq!(us("COVID-19"), "COVID nineteen");
        assert_eq!(us("F1"), "F one");
        assert_eq!(us("3D"), "three D");
    }

    #[test]
    fn whisper_style_sentences() {
        assert_eq!(
            us("I'll pay you $20 by 5pm, OK? That's 50% off."),
            "I'll pay you twenty dollars by five P M, OK? That's fifty percent off."
        );
        assert_eq!(us("We've got 2 hours and 15 minutes left."), "We've got two hours and fifteen minutes left.");
    }

    #[test]
    fn never_panics_on_junk() {
        let mut rng = 0x1234_5678_9abc_def1u64;
        let pool: Vec<char> = "0123456789$£€%:/.-,+#@&x kmKMbBstndrh'\"()ap😀中\n.".chars().collect();
        for len in [1usize, 3, 7, 20, 80] {
            for _ in 0..300 {
                let s: String = (0..len)
                    .map(|_| {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        pool[(rng % pool.len() as u64) as usize]
                    })
                    .collect();
                let _ = normalize(&s, false);
                let _ = normalize(&s, true);
            }
        }
        let big = "9".repeat(200);
        assert!(!normalize(&big, false).is_empty());
        assert!(!normalize(&format!("${big}"), false).is_empty());
    }
}
