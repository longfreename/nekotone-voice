//! English grapheme-to-phoneme conversion into Kokoro's phoneme set
//! (owner: the G2P package; contract fixed by the speak package).
//!
//! Follows the conventions of misaki (hexgrad, Apache-2.0), whose
//! `us_gold.json` / `us_silver.json` (or `gb_*`) lexicons are downloaded
//! with the Kokoro model. No GPL code (no espeak-ng): out-of-vocabulary
//! words go through morphology (s/ed/ing and other suffixes, prefixes,
//! compounds), acronym spelling, and finally letter-to-sound rules
//! ([`super::lts`]).
//!
//! Output conventions (what Kokoro was trained on, misaki `version=None`):
//! words separated by single spaces; stress marks `ˈ` / `ˌ` placed before
//! the stressed vowel; US flap `ɾ` written `T` and glottal stop `ʔ` written
//! `t`; punctuation `;:,.!?—…"()“”` kept, attached to the word before it
//! the way the text had it (`"Hello, world."` → `həlˈO, wˈɜɹld.`).
//!
//! What is ported from misaki's `en.py`: the lexicon lookup with
//! capitalisation variants (`grow_dictionary`), `apply_stress`, the special
//! cases (a, an, am, the, to, in, I, used, by, symbols, dotted
//! abbreviations), `stem_s` / `stem_ed` / `stem_ing`, acronym spelling
//! (`get_NNP`), the right-to-left context (does the next word start with a
//! vowel sound, is it "to") and the stress rules for hyphenated words.
//! spaCy's part-of-speech tags are replaced by a few word-order heuristics
//! (after "to"/modals/subject pronouns → verb; after "have" → past
//! participle; after a determiner → noun; before punctuation → the
//! lexicon's "None" form), and misaki's neural fallback by extra
//! morphology, compound splitting and [`super::lts`].

use crate::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

/// Every symbol Kokoro-82M v1.0 has a token for (from its tokenizer.json,
/// without the pad `$`).
const VOCAB: &str = ";:,.!?\u{2014}\u{2026}\u{22}()\u{201c}\u{201d} \u{303}ʣʥʦʨᵝ\u{ab67}AIOQSTWYᵊabcdefhijklmnopqrstuvwxyzɑɐɒæβɔɕçɖðʤəɚɛɜɟɡɥɨɪʝɯɰŋɳɲɴøɸθœɹɾɻʁɽʂʃʈʧʊʋʌɣɤχʎʒʔˈˌːʰʲ\u{2193}\u{2192}\u{2197}\u{2198}ᵻ";

const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const CONSONANTS: &str = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const US_TAUS: &str = "AIOWYiuæɑəɛɪɹʊʌ";
const DIPHTHONGS: &str = "AIOQWYʤʧ";
const NON_QUOTE_PUNCTS: &str = ";:,.!?—…";
const PRIMARY: char = 'ˈ';
const SECONDARY: char = 'ˌ';

fn is_vowel(c: char) -> bool {
    VOWELS.contains(c)
}

fn symbol_word(s: &str) -> Option<&'static str> {
    Some(match s {
        "%" => "percent",
        "&" => "and",
        "+" => "plus",
        "@" => "at",
        _ => return None,
    })
}

/// One lexicon value: a pronunciation, or pronunciations by part of speech
/// (`DEFAULT`, `NOUN`, `VERB`, `ADJ`, `VBD`, `None`…; a `null` value means
/// "spell it").
#[derive(Clone, Debug)]
enum Entry {
    One(Box<str>),
    Pos(Vec<(Box<str>, Option<Box<str>>)>),
}

impl Entry {
    fn get(&self, tag: &str) -> Option<Option<&str>> {
        match self {
            Entry::One(_) => None,
            Entry::Pos(v) => v.iter().find(|(k, _)| &**k == tag).map(|(_, v)| v.as_deref()),
        }
    }
    fn has(&self, tag: &str) -> bool {
        self.get(tag).is_some()
    }
}

impl<'de> serde::Deserialize<'de> for Entry {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = Entry;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a pronunciation or a map of pronunciations")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> std::result::Result<Entry, E> {
                Ok(Entry::One(v.into()))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut m: A) -> std::result::Result<Entry, A::Error> {
                let mut out = Vec::new();
                while let Some((k, v)) = m.next_entry::<String, Option<String>>()? {
                    out.push((k.into_boxed_str(), v.map(String::into_boxed_str)));
                }
                Ok(Entry::Pos(out))
            }
        }
        d.deserialize_any(V)
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c.flat_map(|x| x.to_lowercase())).collect(),
        None => String::new(),
    }
}

fn is_upper(s: &str) -> bool {
    s == s.to_uppercase()
}

fn is_lower(s: &str) -> bool {
    s == s.to_lowercase()
}

fn is_alpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(char::is_alphabetic)
}

/// misaki's `grow_dictionary`: "word" also answers "Word", "Word" also
/// answers "word"; existing keys win.
fn grow(d: &mut HashMap<String, Entry>) {
    let mut add = Vec::new();
    for (k, v) in d.iter() {
        if k.chars().count() < 2 {
            continue;
        }
        let lower = k.to_lowercase();
        if *k == lower {
            let cap = capitalize(k);
            if cap != *k {
                add.push((cap, v.clone()));
            }
        } else if *k == capitalize(&lower) {
            add.push((lower, v.clone()));
        }
    }
    for (k, v) in add {
        d.entry(k).or_insert(v);
    }
}

/// misaki's `apply_stress`. `stress`: < -1 remove all; -1 (or 0/-0.5 when
/// there is a primary) demote; 0/0.5/1 add a secondary to an unstressed
/// word; ≥ 1 promote a lone secondary; > 1 add a primary.
fn apply_stress(ps: &str, stress: Option<f32>) -> String {
    let Some(stress) = stress else { return ps.to_string() };
    let has_p = ps.contains(PRIMARY);
    let has_any = has_p || ps.contains(SECONDARY);
    let has_vowel = ps.chars().any(is_vowel);
    if stress < -1.0 {
        ps.chars().filter(|&c| c != PRIMARY && c != SECONDARY).collect()
    } else if stress == -1.0 || ((stress == 0.0 || stress == -0.5) && has_p) {
        ps.chars().filter(|&c| c != SECONDARY).map(|c| if c == PRIMARY { SECONDARY } else { c }).collect()
    } else if (stress == 0.0 || stress == 0.5 || stress == 1.0) && !has_any {
        if !has_vowel {
            return ps.to_string();
        }
        restress(&format!("{SECONDARY}{ps}"))
    } else if stress >= 1.0 && !has_p && ps.contains(SECONDARY) {
        ps.replace(SECONDARY, &PRIMARY.to_string())
    } else if stress > 1.0 && !has_any {
        if !has_vowel {
            return ps.to_string();
        }
        restress(&format!("{PRIMARY}{ps}"))
    } else {
        ps.to_string()
    }
}

/// Move every stress mark to just before the next vowel.
fn restress(ps: &str) -> String {
    let chars: Vec<char> = ps.chars().collect();
    let mut keyed: Vec<(f64, char)> = chars.iter().enumerate().map(|(i, &c)| (i as f64, c)).collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == PRIMARY || c == SECONDARY {
            if let Some(j) = (i..chars.len()).find(|&j| is_vowel(chars[j])) {
                keyed[i].0 = j as f64 - 0.5;
            }
        }
    }
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    keyed.into_iter().map(|(_, c)| c).collect()
}

fn stress_weight(ps: &str) -> usize {
    ps.chars().map(|c| if DIPHTHONGS.contains(c) { 2 } else { 1 }).sum()
}

fn last_char(s: &str) -> Option<char> {
    s.chars().last()
}

fn nth_last(s: &str, n: usize) -> Option<char> {
    s.chars().rev().nth(n)
}

fn drop_last(s: &str) -> &str {
    match s.char_indices().last() {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Context from the words to the right (misaki's `TokenContext`).
#[derive(Clone, Copy, Debug, Default)]
struct Ctx {
    /// Does the next sound start with a vowel? None: punctuation or the end.
    future_vowel: Option<bool>,
    /// Is the next word "to"?
    future_to: bool,
}

fn parent_tag(tag: Option<&str>) -> Option<&str> {
    let t = tag?;
    Some(if t.starts_with("VB") {
        "VERB"
    } else if t.starts_with("NN") {
        "NOUN"
    } else if t.starts_with("ADV") || t.starts_with("RB") {
        "ADV"
    } else if t.starts_with("ADJ") || t.starts_with("JJ") {
        "ADJ"
    } else {
        t
    })
}

/// A pronunciation dictionary (gold entries first, then silver).
pub struct Lexicon {
    british: bool,
    gold: HashMap<String, Entry>,
    silver: HashMap<String, Entry>,
}

impl Lexicon {
    /// Read `{us|gb}_gold.json` and `{us|gb}_silver.json` from `dir`.
    /// Errors are sentences naming the missing or unreadable file.
    pub fn load(dir: &Path, british: bool) -> Result<Lexicon> {
        let p = if british { "gb" } else { "us" };
        let read = |name: String| -> Result<String> {
            let path = dir.join(&name);
            std::fs::read_to_string(&path).map_err(|e| {
                Error::Model(format!(
                    "could not read the pronunciation dictionary {} ({e}); download the Kokoro voice model again in Settings → Models",
                    path.display()
                ))
            })
        };
        let gold = read(format!("{p}_gold.json"))?;
        let silver = read(format!("{p}_silver.json"))?;
        Lexicon::from_json(&gold, &silver, british)
    }

    /// Build from the JSON text of the two files (tests use small maps).
    pub fn from_json(gold: &str, silver: &str, british: bool) -> Result<Lexicon> {
        let parse = |s: &str, what: &str| -> Result<HashMap<String, Entry>> {
            serde_json::from_str(s).map_err(|e| Error::Model(format!("the {what} pronunciation dictionary is damaged ({e}); download the voice model again")))
        };
        let mut gold = parse(gold, "gold")?;
        let mut silver = parse(silver, "silver")?;
        grow(&mut gold);
        grow(&mut silver);
        Ok(Lexicon { british, gold, silver })
    }

    pub fn british(&self) -> bool {
        self.british
    }

    fn gold_str(&self, w: &str) -> Option<&str> {
        match self.gold.get(w)? {
            Entry::One(s) => Some(s),
            Entry::Pos(_) => self.gold.get(w).and_then(|e| e.get("DEFAULT")).flatten(),
        }
    }

    fn contains(&self, w: &str) -> bool {
        self.gold.contains_key(w) || self.silver.contains_key(w)
    }

    /// Letters spelled one by one, last one stressed (misaki `get_NNP`).
    fn get_nnp(&self, word: &str) -> Option<String> {
        let mut ps = String::new();
        let mut any = false;
        for c in word.chars().filter(|c| c.is_alphabetic()) {
            let up: String = c.to_uppercase().collect();
            ps.push_str(self.gold_str(&up)?);
            any = true;
        }
        if !any {
            return None;
        }
        let ps = apply_stress(&ps, Some(0.0));
        match ps.rfind(SECONDARY) {
            Some(i) => Some(format!("{}{PRIMARY}{}", &ps[..i], &ps[i + SECONDARY.len_utf8()..])),
            None => Some(ps),
        }
    }

    fn is_known(&self, word: &str) -> bool {
        if self.contains(word) || symbol_word(word).is_some() {
            return true;
        }
        if !is_alpha(word) || !word.chars().all(|c| c.is_ascii_alphabetic()) {
            return false;
        }
        if word.chars().count() == 1 {
            return true;
        }
        if is_upper(word) && self.gold.contains_key(&word.to_lowercase()) {
            return true;
        }
        let rest: String = word.chars().skip(1).collect();
        rest == rest.to_uppercase()
    }

    fn lookup(&self, word: &str, tag: Option<&str>, stress: Option<f32>, ctx: Option<&Ctx>) -> Option<String> {
        let mut word = word.to_string();
        let mut is_nnp = false;
        if is_upper(&word) && !self.gold.contains_key(&word) {
            word = word.to_lowercase();
            is_nnp = tag == Some("NNP");
        }
        let mut entry = self.gold.get(&word);
        if entry.is_none() && !is_nnp {
            entry = self.silver.get(&word);
        }
        let ps: Option<String> = match entry {
            None => None,
            Some(Entry::One(s)) => Some(s.to_string()),
            Some(e @ Entry::Pos(_)) => {
                let t: Option<&str> = if ctx.map(|c| c.future_vowel.is_none()).unwrap_or(false) && e.has("None") {
                    Some("None")
                } else if tag.map(|t| e.has(t)).unwrap_or(false) {
                    tag
                } else {
                    parent_tag(tag)
                };
                match t.and_then(|t| e.get(t)) {
                    Some(v) => v.map(str::to_string),
                    None => e.get("DEFAULT").flatten().map(str::to_string),
                }
            }
        };
        if ps.is_none() || (is_nnp && !ps.as_deref().unwrap_or("").contains(PRIMARY)) {
            if let Some(n) = self.get_nnp(&word) {
                return Some(n);
            }
        }
        ps.map(|p| apply_stress(&p, stress))
    }

    fn special_case(&self, word: &str, tag: Option<&str>, stress: Option<f32>, ctx: &Ctx) -> Option<String> {
        if let Some(s) = symbol_word(word) {
            return self.lookup(s, None, None, Some(ctx));
        }
        let stripped = word.trim_matches('.');
        if stripped.contains('.') && is_alpha(&word.replace('.', "")) && word.split('.').map(|p| p.chars().count()).max().unwrap_or(0) < 3 {
            return self.get_nnp(word);
        }
        match word {
            "a" | "A" => return Some(if tag == Some("DT") { "ɐ".into() } else { "ˈA".into() }),
            "am" | "Am" | "AM" => {
                if tag.map(|t| t.starts_with("NN")).unwrap_or(false) {
                    return self.get_nnp(word);
                } else if ctx.future_vowel.is_none() || word != "am" || stress.map(|s| s > 0.0).unwrap_or(false) {
                    return self.gold_str("am").map(str::to_string);
                }
                return Some("ɐm".into());
            }
            "an" | "An" | "AN" => {
                if word == "AN" && tag.map(|t| t.starts_with("NN")).unwrap_or(false) {
                    return self.get_nnp(word);
                }
                return Some("ɐn".into());
            }
            "I" if tag == Some("PRP") => return Some(format!("{SECONDARY}I")),
            "by" | "By" | "BY" if parent_tag(tag) == Some("ADV") => return Some("bˈI".into()),
            "to" | "To" => return Some(to_form(self, ctx)),
            "TO" if matches!(tag, Some("TO") | Some("IN")) => return Some(to_form(self, ctx)),
            "in" | "In" => {
                let s = if ctx.future_vowel.is_none() || tag != Some("IN") { "ˈ" } else { "" };
                return Some(format!("{s}ɪn"));
            }
            "IN" if tag != Some("NNP") => {
                let s = if ctx.future_vowel.is_none() || tag != Some("IN") { "ˈ" } else { "" };
                return Some(format!("{s}ɪn"));
            }
            "the" | "The" => return Some(if ctx.future_vowel == Some(true) { "ði".into() } else { "ðə".into() }),
            "THE" if tag == Some("DT") => return Some(if ctx.future_vowel == Some(true) { "ði".into() } else { "ðə".into() }),
            "vs" | "vs." | "Vs" | "VS" if tag == Some("IN") => return self.lookup("versus", None, None, Some(ctx)),
            "used" | "Used" | "USED" => {
                let key = if matches!(tag, Some("VBD") | Some("JJ")) && ctx.future_to { "VBD" } else { "DEFAULT" };
                return self.gold.get("used").and_then(|e| e.get(key)).flatten().map(str::to_string);
            }
            _ => {}
        }
        return None;

        fn to_form(lex: &Lexicon, ctx: &Ctx) -> String {
            match ctx.future_vowel {
                None => lex.gold_str("to").unwrap_or("tʊ").to_string(),
                Some(false) => "tə".into(),
                Some(true) => "tʊ".into(),
            }
        }
    }

    fn suffix_s(&self, stem: &str) -> Option<String> {
        let last = last_char(stem)?;
        Some(if "ptkfθ".contains(last) {
            format!("{stem}s")
        } else if "szʃʒʧʤ".contains(last) {
            format!("{stem}{}z", if self.british { 'ɪ' } else { 'ᵻ' })
        } else {
            format!("{stem}z")
        })
    }

    fn stem_s(&self, word: &str, tag: Option<&str>, stress: Option<f32>, ctx: &Ctx) -> Option<String> {
        let n = word.chars().count();
        if n < 3 || !word.ends_with('s') {
            return None;
        }
        let stem = if !word.ends_with("ss") && self.is_known(&word[..word.len() - 1]) {
            word[..word.len() - 1].to_string()
        } else if (word.ends_with("'s") || (n > 4 && word.ends_with("es") && !word.ends_with("ies"))) && self.is_known(&word[..word.len() - 2]) {
            word[..word.len() - 2].to_string()
        } else if n > 4 && word.ends_with("ies") && self.is_known(&format!("{}y", &word[..word.len() - 3])) {
            format!("{}y", &word[..word.len() - 3])
        } else {
            return None;
        };
        let ps = self.lookup(&stem, tag, stress, Some(ctx))?;
        self.suffix_s(&ps)
    }

    fn suffix_ed(&self, stem: &str) -> Option<String> {
        let last = last_char(stem)?;
        let r = if self.british { 'ɪ' } else { 'ᵻ' };
        Some(if "pkfθʃsʧ".contains(last) {
            format!("{stem}t")
        } else if last == 'd' {
            format!("{stem}{r}d")
        } else if last != 't' {
            format!("{stem}d")
        } else if self.british || stem.chars().count() < 2 {
            format!("{stem}ɪd")
        } else if nth_last(stem, 1).map(|c| US_TAUS.contains(c)).unwrap_or(false) {
            format!("{}ɾᵻd", drop_last(stem))
        } else {
            format!("{stem}ᵻd")
        })
    }

    fn stem_ed(&self, word: &str, tag: Option<&str>, stress: Option<f32>, ctx: &Ctx) -> Option<String> {
        let n = word.chars().count();
        if n < 4 || !word.ends_with('d') {
            return None;
        }
        let stem = if !word.ends_with("dd") && self.is_known(&word[..word.len() - 1]) {
            &word[..word.len() - 1]
        } else if n > 4 && word.ends_with("ed") && !word.ends_with("eed") && self.is_known(&word[..word.len() - 2]) {
            &word[..word.len() - 2]
        } else {
            return None;
        };
        let ps = self.lookup(stem, tag, stress, Some(ctx))?;
        self.suffix_ed(&ps)
    }

    fn suffix_ing(&self, stem: &str) -> Option<String> {
        let last = last_char(stem)?;
        if self.british {
            if "əː".contains(last) {
                return None;
            }
        } else if stem.chars().count() > 1 && last == 't' && nth_last(stem, 1).map(|c| US_TAUS.contains(c)).unwrap_or(false) {
            return Some(format!("{}ɾɪŋ", drop_last(stem)));
        }
        Some(format!("{stem}ɪŋ"))
    }

    fn stem_ing(&self, word: &str, tag: Option<&str>, stress: Option<f32>, ctx: &Ctx) -> Option<String> {
        let n = word.chars().count();
        if n < 5 || !word.ends_with("ing") || !word.is_ascii() {
            return None;
        }
        let base = &word[..word.len() - 3];
        let stem = if n > 5 && self.is_known(base) {
            base.to_string()
        } else if self.is_known(&format!("{base}e")) {
            format!("{base}e")
        } else if n > 5 && doubled_before_ing(word) && self.is_known(&word[..word.len() - 4]) {
            word[..word.len() - 4].to_string()
        } else {
            return None;
        };
        let ps = self.lookup(&stem, tag, stress, Some(ctx))?;
        self.suffix_ing(&ps)
    }

    /// misaki `get_word`: special cases, the lexicon, then -s/-ed/-ing on a known stem.
    fn get_word(&self, word: &str, tag: Option<&str>, stress: Option<f32>, ctx: &Ctx) -> Option<String> {
        if let Some(ps) = self.special_case(word, tag, stress, ctx) {
            return Some(ps);
        }
        let wl = word.to_lowercase();
        let mut word = word.to_string();
        let rest: String = word.chars().skip(1).collect();
        if word.chars().count() > 1
            && is_alpha(&word.replace('\'', ""))
            && word != wl
            && (tag != Some("NNP") || word.chars().count() > 7)
            && !self.contains(&word)
            && (is_upper(&word) || is_lower(&rest))
            && (self.contains(&wl)
                || self.stem_s(&wl, tag, stress, ctx).is_some()
                || self.stem_ed(&wl, tag, stress, ctx).is_some()
                || self.stem_ing(&wl, tag, stress, ctx).is_some())
        {
            word = wl;
        }
        if self.is_known(&word) {
            return self.lookup(&word, tag, stress, Some(ctx));
        }
        if word.ends_with("s'") && self.is_known(&format!("{}'s", &word[..word.len() - 2])) {
            return self.lookup(&format!("{}'s", &word[..word.len() - 2]), tag, stress, Some(ctx));
        }
        if word.ends_with('\'') && self.is_known(&word[..word.len() - 1]) {
            return self.lookup(&word[..word.len() - 1], tag, stress, Some(ctx));
        }
        if let Some(p) = self.stem_s(&word, tag, stress, ctx) {
            return Some(p);
        }
        if let Some(p) = self.stem_ed(&word, tag, stress, ctx) {
            return Some(p);
        }
        self.stem_ing(&word, tag, Some(stress.unwrap_or(0.5)), ctx)
    }

    /// misaki `Lexicon.__call__` for a word without digits.
    fn call(&self, word: &str, tag: Option<&str>, ctx: &Ctx) -> Option<String> {
        let stress = if is_lower(word) {
            None
        } else if is_upper(word) {
            Some(2.0)
        } else {
            Some(0.5)
        };
        self.get_word(word, tag, stress, ctx)
    }
}

fn doubled_before_ing(word: &str) -> bool {
    let b = word.as_bytes();
    if b.len() < 5 {
        return false;
    }
    let (x, y) = (b[b.len() - 5], b[b.len() - 4]);
    (x == y && b"bcdgklmnprstvxz".contains(&x)) || word.ends_with("cking")
}

/// Suffixes the extra morphology knows: (suffix, US phonemes, GB phonemes).
const SUFFIXES: &[(&str, &str, &str)] = &[
    ("ness", "nəs", "nəs"),
    ("ment", "mənt", "mənt"),
    ("less", "ləs", "ləs"),
    ("ful", "fəl", "fəl"),
    ("able", "əbᵊl", "əbᵊl"),
    ("ible", "əbᵊl", "ɪbᵊl"),
    ("ably", "əbli", "əbli"),
    ("ly", "li", "li"),
    ("ers", "əɹz", "əz"),
    ("er", "əɹ", "ə"),
    ("est", "əst", "ɪst"),
    ("ish", "ɪʃ", "ɪʃ"),
    ("ism", "ˌɪzəm", "ˌɪzəm"),
    ("ist", "ɪst", "ɪst"),
    ("ists", "ɪsts", "ɪsts"),
    ("ize", "ˌIz", "ˌIz"),
    ("ise", "ˌIz", "ˌIz"),
    ("hood", "hˌʊd", "hˌʊd"),
    ("ship", "ʃˌɪp", "ʃˌɪp"),
    ("dom", "dəm", "dəm"),
    ("wards", "wəɹdz", "wədz"),
    ("ward", "wəɹd", "wəd"),
    ("wise", "wˌIz", "wˌIz"),
    ("like", "lˌIk", "lˌIk"),
    ("y", "i", "i"),
];

/// Prefixes: (prefix, US, GB). The stem keeps its primary stress.
const PREFIXES: &[(&str, &str, &str)] = &[
    ("counter", "kˌWntəɹ", "kˌWntə"),
    ("under", "ˌʌndəɹ", "ˌʌndə"),
    ("over", "ˌOvəɹ", "ˌQvə"),
    ("inter", "ˌɪntəɹ", "ˌɪntə"),
    ("super", "sˌupəɹ", "sˌuːpə"),
    ("hyper", "hˌIpəɹ", "hˌIpə"),
    ("cyber", "sˌIbəɹ", "sˌIbə"),
    ("ultra", "ˌʌltɹə", "ˌʌltɹə"),
    ("micro", "mˌIkɹO", "mˌIkɹQ"),
    ("multi", "mˌʌlti", "mˌʌlti"),
    ("trans", "tɹˌænz", "tɹˌanz"),
    ("anti", "ˌænti", "ˌanti"),
    ("semi", "sˌɛmi", "sˌɛmi"),
    ("mega", "mˌɛɡə", "mˌɛɡə"),
    ("auto", "ˌɔɾO", "ˌɔːtQ"),
    ("post", "pˌOst", "pˌQst"),
    ("fore", "fˌɔɹ", "fˌɔː"),
    ("self", "sˌɛlf", "sˌɛlf"),
    ("non", "nˌɑn", "nˌɒn"),
    ("mis", "mˌɪs", "mˌɪs"),
    ("dis", "dˌɪs", "dˌɪs"),
    ("pre", "pɹˌi", "pɹˌiː"),
    ("sub", "sˌʌb", "sˌʌb"),
    ("out", "ˌWt", "ˌWt"),
    ("mid", "mˌɪd", "mˌɪd"),
    ("bio", "bˌIO", "bˌIQ"),
    ("eco", "ˌikO", "ˌiːkQ"),
    ("un", "ˌʌn", "ˌʌn"),
    ("re", "ɹˌi", "ɹˌiː"),
    ("co", "kˌO", "kˌQ"),
    ("de", "dˌi", "dˌiː"),
];

/// A word or punctuation token of the input.
#[derive(Debug, Clone)]
struct Tok {
    text: String,
    word: bool,
    space: bool,
}

const VERB_TRIGGERS: &[&str] = &[
    "to", "will", "would", "can", "could", "should", "must", "might", "may", "shall", "do", "does", "did", "don't", "doesn't", "didn't", "won't",
    "can't", "cannot", "couldn't", "shouldn't", "wouldn't", "let's", "please", "i", "you", "we", "they", "i'll", "you'll", "we'll", "they'll",
    "never", "always", "often", "usually", "gonna", "wanna", "he", "she", "who", "it",
];
const PERFECT: &[&str] = &["has", "have", "had", "i've", "you've", "we've", "they've", "having", "hasn't", "haven't", "hadn't", "i'd", "you'd", "we'd", "they'd"];
const BE: &[&str] = &["is", "are", "was", "were", "be", "been", "being", "am", "isn't", "aren't", "wasn't", "weren't", "it's", "that's", "he's", "she's"];
const DETERMINERS: &[&str] = &[
    "a", "an", "the", "my", "your", "his", "her", "its", "our", "their", "this", "that", "these", "those", "some", "any", "no", "every", "each",
    "another", "first", "last", "next",
];

/// Word-order guess standing in for spaCy's tagger.
fn guess_tag(prev: Option<&str>, word: &str, next_is_word: bool) -> Option<&'static str> {
    match word {
        "a" | "A" => return if next_is_word { Some("DT") } else { None },
        "I" => return Some("PRP"),
        "in" | "In" | "IN" => return Some("IN"),
        "the" | "The" | "THE" => return Some("DT"),
        "to" | "To" | "TO" => return Some("TO"),
        "used" | "Used" | "USED" => return Some("VBD"),
        "that" | "That" if !next_is_word => return Some("DT"),
        _ => {}
    }
    let prev = prev?.to_lowercase().replace('’', "'");
    let p = prev.as_str();
    if VERB_TRIGGERS.contains(&p) {
        Some("VERB")
    } else if PERFECT.contains(&p) {
        Some("VBN")
    } else if BE.contains(&p) {
        Some("ADJ")
    } else if DETERMINERS.contains(&p) {
        Some("NOUN")
    } else {
        None
    }
}

/// Text → phonemes.
pub struct G2p {
    lexicon: Lexicon,
}

impl G2p {
    pub fn new(lexicon: Lexicon) -> G2p {
        G2p { lexicon }
    }

    pub fn british(&self) -> bool {
        self.lexicon.british()
    }

    /// Phonemes of normalised text (see [`super::normalize::normalize`]):
    /// words, spaces and sentence punctuation. Characters outside Kokoro's
    /// vocabulary never appear in the result. Never panics; unknown words
    /// always get *some* pronunciation.
    pub fn phonemize(&self, text: &str) -> String {
        let raw = self.phonemize_raw(text, 0);
        finish(&raw)
    }

    /// Pronunciation of one word, as it would appear in [`phonemize`]
    /// with no context (for tests and the voice gallery). None only for an
    /// empty or all-punctuation word.
    pub fn word(&self, word: &str) -> Option<String> {
        let toks = self.tokenize(word);
        if !toks.iter().any(|t| t.word) {
            return None;
        }
        let s = self.phonemize(word);
        let s: String = s.chars().filter(|c| !NON_QUOTE_PUNCTS.contains(*c) && !"\"“”()".contains(*c)).collect();
        let s = s.trim().to_string();
        (!s.is_empty()).then_some(s)
    }

    fn phonemize_raw(&self, text: &str, depth: usize) -> String {
        let toks = self.tokenize(text);
        let n = toks.len();
        let mut out: Vec<String> = vec![String::new(); n];
        let mut ctx = Ctx::default();
        for i in (0..n).rev() {
            let t = &toks[i];
            if !t.word {
                out[i] = t.text.clone();
                if let Some(c) = t.text.chars().find(|c| is_vowel(*c) || CONSONANTS.contains(*c) || NON_QUOTE_PUNCTS.contains(*c)) {
                    if NON_QUOTE_PUNCTS.contains(c) {
                        ctx.future_vowel = None;
                    }
                }
                ctx.future_to = false;
                continue;
            }
            let prev = toks[..i].iter().rev().find(|p| p.word).filter(|_| {
                // only a word directly before (no punctuation in between)
                i > 0 && toks[i - 1].word
            });
            let next_is_word = toks.get(i + 1).map(|t| t.word).unwrap_or(false);
            let tag = guess_tag(prev.map(|p| p.text.as_str()), &t.text, next_is_word);
            let ps = self.word_ps(&t.text, tag, &ctx, depth);
            // update the context for the word to the left
            if let Some(c) = ps.chars().find(|c| is_vowel(*c) || CONSONANTS.contains(*c) || NON_QUOTE_PUNCTS.contains(*c)) {
                ctx.future_vowel = if NON_QUOTE_PUNCTS.contains(c) { None } else { Some(is_vowel(c)) };
            }
            ctx.future_to = matches!(t.text.as_str(), "to" | "To") || (t.text == "TO" && matches!(tag, Some("TO") | Some("IN")));
            out[i] = ps;
        }
        let mut s = String::new();
        for (i, t) in toks.iter().enumerate() {
            s.push_str(&out[i]);
            if t.space {
                s.push(' ');
            }
        }
        s
    }

    fn tokenize(&self, text: &str) -> Vec<Tok> {
        let chars: Vec<char> = text.chars().map(|c| if c == '’' || c == '‘' || c == 'ʼ' { '\'' } else { c }).collect();
        let n = chars.len();
        let mut toks: Vec<Tok> = Vec::new();
        let is_wc = |c: char| c.is_alphanumeric();
        let mut i = 0;
        while i < n {
            let c = chars[i];
            if c.is_whitespace() {
                if let Some(t) = toks.last_mut() {
                    t.space = true;
                }
                i += 1;
                continue;
            }
            let prev_c = if i > 0 { Some(chars[i - 1]) } else { None };
            let starts_word = is_wc(c) || (c == '\'' && i + 1 < n && chars[i + 1].is_alphabetic() && !prev_c.map(is_wc).unwrap_or(false) && {
                // a leading apostrophe the lexicon knows ('cause, 'em, 'til)
                let mut j = i + 1;
                while j < n && chars[j].is_alphabetic() {
                    j += 1;
                }
                let w: String = chars[i..j].iter().collect();
                self.lexicon.contains(&w) || self.lexicon.contains(&w.to_lowercase())
            });
            if starts_word {
                let start = i;
                i += 1;
                while i < n {
                    let c = chars[i];
                    let p = chars[i - 1];
                    let next = chars.get(i + 1).copied();
                    let ok = is_wc(c)
                        || (c == '\'' && is_wc(p) && next.map(char::is_alphabetic).unwrap_or(false))
                        || (c == '\'' && (p == 's' || p == 'S') && !next.map(is_wc).unwrap_or(false))
                        || (c == '-' && is_wc(p) && next.map(is_wc).unwrap_or(false))
                        || (c == '.' && p.is_alphabetic() && next.map(char::is_alphabetic).unwrap_or(false) && {
                            // U.S.A / e.g — single letters between the dots
                            let before = chars[start..i].iter().rev().take_while(|c| c.is_alphabetic()).count();
                            let after = chars[i + 1..].iter().take_while(|c| c.is_alphabetic()).count();
                            before <= 2 && after <= 2
                        })
                        || ((c == '.' || c == ',') && p.is_ascii_digit() && next.map(|x| x.is_ascii_digit()).unwrap_or(false));
                    if !ok {
                        break;
                    }
                    i += 1;
                }
                let w: String = chars[start..i].iter().collect();
                toks.push(Tok { text: w, word: true, space: false });
                continue;
            }
            let prev_space = i == 0 || chars[i - 1].is_whitespace() || "([{".contains(chars[i - 1]);
            let p: Option<String> = match c {
                ';' | ':' | ',' | '!' | '?' | '(' | ')' | '—' | '…' | '“' | '”' => Some(c.to_string()),
                '.' => {
                    if i + 2 < n && chars[i + 1] == '.' && chars[i + 2] == '.' {
                        i += 2;
                        Some("…".into())
                    } else {
                        Some(".".into())
                    }
                }
                '[' | '{' => Some("(".into()),
                ']' | '}' => Some(")".into()),
                '–' | '―' | '‒' => Some("—".into()),
                '-' => {
                    if i + 1 < n && chars[i + 1] == '-' {
                        i += 1;
                    }
                    Some("—".into())
                }
                '"' | '\'' | '«' | '»' | '„' => Some(if prev_space { "“".into() } else { "”".into() }),
                '%' | '&' | '+' | '@' => {
                    toks.push(Tok { text: c.to_string(), word: true, space: false });
                    None
                }
                _ => None,
            };
            if let Some(p) = p {
                toks.push(Tok { text: p, word: false, space: false });
            }
            i += 1;
        }
        toks
    }

    /// Phonemes of one word token (may hold digits, hyphens, dots, apostrophes).
    fn word_ps(&self, text: &str, tag: Option<&str>, ctx: &Ctx, depth: usize) -> String {
        let lex = &self.lexicon;
        if text.chars().any(|c| c.is_ascii_digit()) {
            return self.number_ps(text, depth);
        }
        if text.contains('-') {
            if let Some(p) = lex.call(text, tag, ctx) {
                return p;
            }
            let parts: Vec<&str> = text.split('-').filter(|p| !p.is_empty()).collect();
            let mut ps: Vec<String> = parts.iter().map(|p| self.word_ps(p, None, ctx, depth)).collect();
            resolve_compound(&parts, &mut ps);
            return ps.concat();
        }
        // An all-caps word the lexicon says to spell as a noun ("US").
        if text.chars().count() >= 2 && is_upper(text) && is_alpha(text) {
            if let Some(e) = lex.gold.get(text) {
                if e.get("NOUN") == Some(None) {
                    if let Some(p) = lex.get_nnp(text) {
                        return p;
                    }
                }
            }
        }
        if let Some(p) = lex.call(text, tag, ctx) {
            return p;
        }
        self.oov(text, depth)
    }

    fn number_ps(&self, text: &str, depth: usize) -> String {
        if depth < 2 {
            let words = super::normalize::normalize(text, self.british());
            if !words.chars().any(|c| c.is_ascii_digit()) && words.trim() != text {
                let s = self.phonemize_raw(&words, depth + 1);
                return s.chars().filter(|c| !NON_QUOTE_PUNCTS.contains(*c)).collect::<String>().trim().to_string();
            }
        }
        // digit by digit, letters spelled
        const DIGITS: [&str; 10] = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
        let mut parts = Vec::new();
        for c in text.chars() {
            if let Some(d) = c.to_digit(10) {
                if let Some(p) = self.lexicon.lookup(DIGITS[d as usize], None, None, None) {
                    parts.push(p);
                }
            } else if c.is_alphabetic() {
                if let Some(p) = self.lexicon.get_nnp(&c.to_string()) {
                    parts.push(p);
                }
            }
        }
        parts.join(" ")
    }

    /// Words neither the lexicon nor misaki's stemming knows.
    fn oov(&self, text: &str, depth: usize) -> String {
        let lex = &self.lexicon;
        let folded = fold_accents(text);
        if folded != text && !folded.is_empty() {
            if let Some(p) = lex.call(&folded, None, &Ctx::default()) {
                return p;
            }
        }
        let w: String = folded.chars().filter(|c| c.is_ascii_alphabetic() || *c == '\'' || *c == '.').collect();
        let w = w.trim_matches('.').replace('.', "");
        if w.is_empty() {
            return String::new();
        }
        // contractions of unknown words
        if let Some(pos) = w.rfind('\'') {
            let (base, suf) = w.split_at(pos);
            let low = suf.to_lowercase();
            if !base.is_empty() {
                if base.to_lowercase().ends_with('n') && low == "'t" {
                    let b = &base[..base.len() - 1];
                    if !b.is_empty() {
                        return format!("{}ᵊnt", self.word_ps(b, None, &Ctx::default(), depth));
                    }
                }
                let bp = self.word_ps(base, None, &Ctx::default(), depth);
                let r = match low.as_str() {
                    "'s" => lex.suffix_s(&bp),
                    "'ll" => Some(format!("{bp}əl")),
                    "'re" => Some(format!("{bp}{}", if self.british() { "ə" } else { "əɹ" })),
                    "'ve" => Some(format!("{bp}v")),
                    "'d" => Some(format!("{bp}d")),
                    "'m" => Some(format!("{bp}m")),
                    _ => None,
                };
                if let Some(r) = r {
                    return r;
                }
            }
            let joined = w.replace('\'', "");
            return self.oov(&joined, depth);
        }
        let n = w.chars().count();
        let lower = w.to_lowercase();
        let no_vowel = !lower.chars().any(|c| "aeiouy".contains(c));
        if ((2..=6).contains(&n) && is_upper(&w)) || (no_vowel && n <= 5) {
            if let Some(p) = lex.get_nnp(&w) {
                return p;
            }
        }
        if let Some(p) = self.derive(&lower, 0) {
            return p;
        }
        super::lts::letters_to_sounds(&lower, self.british())
    }

    /// A word the lexicon knows, with misaki's -s/-ed/-ing.
    fn known(&self, w: &str) -> Option<String> {
        self.lexicon.get_word(w, None, None, &Ctx::default())
    }

    /// Build an unknown lower-case word from known parts: suffixes,
    /// prefixes, compounds. None when it cannot (then letter-to-sound).
    fn derive(&self, w: &str, depth: usize) -> Option<String> {
        if depth > 2 || !w.is_ascii() || w.len() < 4 {
            return None;
        }
        let br = self.british();
        let resolve = |s: &str| -> Option<String> { self.known(s).or_else(|| self.derive(s, depth + 1)) };
        // -s / -ed / -ing on a derived stem
        if let Some(stem) = w.strip_suffix("ing") {
            for cand in stem_candidates(stem) {
                if let Some(p) = resolve(&cand) {
                    if let Some(r) = self.lexicon.suffix_ing(&p) {
                        return Some(r);
                    }
                }
            }
        }
        if let Some(stem) = w.strip_suffix("ed") {
            for cand in stem_candidates(stem) {
                if let Some(p) = resolve(&cand) {
                    return self.lexicon.suffix_ed(&p);
                }
            }
        }
        if w.ends_with('s') && !w.ends_with("ss") {
            let stems = if let Some(s) = w.strip_suffix("ies") { vec![format!("{s}y")] } else { vec![w[..w.len() - 1].to_string()] };
            for cand in stems.iter().chain(w.strip_suffix("es").map(str::to_string).iter()) {
                if cand.len() >= 3 {
                    if let Some(p) = resolve(cand) {
                        return self.lexicon.suffix_s(&p);
                    }
                }
            }
        }
        for &(suf, us, gb) in SUFFIXES {
            let Some(stem) = w.strip_suffix(suf) else { continue };
            if stem.len() < 3 {
                continue;
            }
            for cand in stem_candidates(stem) {
                if let Some(p) = resolve(&cand) {
                    return Some(join_suffix(&p, &cand, stem, suf, if br { gb } else { us }));
                }
            }
        }
        if let Some(p) = self.compound(w) {
            return Some(p);
        }
        for &(pre, us, gb) in PREFIXES {
            let Some(rest) = w.strip_prefix(pre) else { continue };
            if rest.len() < 3 {
                continue;
            }
            if let Some(p) = resolve(rest) {
                return Some(format!("{}{}", if br { gb } else { us }, p));
            }
        }
        None
    }

    /// Split into two or three known words ("microsoft" → micro + soft);
    /// the first part keeps the primary stress.
    fn compound(&self, w: &str) -> Option<String> {
        let n = w.len();
        if n < 6 {
            return None;
        }
        let lex = &self.lexicon;
        let part = |s: &str| -> Option<String> { if lex.contains(s) { lex.lookup(s, None, None, None) } else { None } };
        let last = |s: &str| -> Option<String> { part(s).or_else(|| self.known(s)) };
        let mut best: Option<(usize, String)> = None;
        for i in 3..=n - 3 {
            let (a, b) = w.split_at(i);
            if a.len().max(b.len()) < 4 {
                continue;
            }
            let score = a.len().min(b.len());
            if best.as_ref().map(|(s, _)| score <= *s).unwrap_or(false) {
                continue;
            }
            if let (Some(pa), Some(pb)) = (part(a), last(b)) {
                best = Some((score, format!("{pa}{}", apply_stress(&pb, Some(-1.0)))));
            }
        }
        if let Some((_, p)) = best {
            return Some(p);
        }
        if n < 9 {
            return None;
        }
        let mut best: Option<(usize, String)> = None;
        for i in 3..=n - 6 {
            for j in i + 3..=n - 3 {
                let (a, b, c) = (&w[..i], &w[i..j], &w[j..]);
                let score = a.len().min(b.len()).min(c.len());
                if best.as_ref().map(|(s, _)| score <= *s).unwrap_or(false) {
                    continue;
                }
                if let (Some(pa), Some(pb), Some(pc)) = (part(a), part(b), last(c)) {
                    best = Some((score, format!("{pa}{}{}", apply_stress(&pb, Some(-1.0)), apply_stress(&pc, Some(-1.0)))));
                }
            }
        }
        best.map(|(_, p)| p)
    }
}

/// Spellings a stem may have had before a suffix: as is, with a silent e,
/// y for i, and an undoubled final consonant.
fn stem_candidates(stem: &str) -> Vec<String> {
    let mut v = vec![stem.to_string(), format!("{stem}e")];
    if let Some(s) = stem.strip_suffix('i') {
        v.push(format!("{s}y"));
    }
    let b = stem.as_bytes();
    if b.len() >= 3 && b[b.len() - 1] == b[b.len() - 2] && !b"aeiou".contains(&b[b.len() - 1]) {
        v.push(stem[..stem.len() - 1].to_string());
    }
    if stem.ends_with("ab") || stem.ends_with("ib") {
        v.push(format!("{stem}le"));
    }
    v
}

fn join_suffix(p: &str, cand: &str, stem: &str, suf: &str, add: &str) -> String {
    // comfortab|ly → comfortable → …bᵊl → …bli
    if suf == "ly" && cand.ends_with("le") && !stem.ends_with("le") {
        for tail in ["ᵊl", "əl", "l"] {
            if let Some(s) = p.strip_suffix(tail) {
                return format!("{s}li");
            }
        }
    }
    // happ|ily → happy → hˈæpi → hˈæpəli
    if suf == "ly" && cand.ends_with('y') && stem.ends_with('i') {
        if let Some(s) = p.strip_suffix('i') {
            return format!("{s}əli");
        }
    }
    // a silent -e stem before a vowel suffix loses nothing in sound
    format!("{p}{add}")
}

/// misaki's `resolve_tokens` for the parts of a hyphenated word: demote
/// all but the heaviest stresses so the word has one main stress.
fn resolve_compound(parts: &[&str], ps: &mut [String]) {
    let mixed = {
        let alpha = parts.iter().any(|p| p.chars().any(char::is_alphabetic));
        let digit = parts.iter().any(|p| p.chars().any(|c| c.is_ascii_digit()));
        alpha && digit
    };
    if mixed {
        for p in ps.iter_mut().skip(1) {
            p.insert(0, ' ');
        }
        return;
    }
    let idx: Vec<(bool, usize, usize)> = ps.iter().enumerate().filter(|(_, p)| !p.is_empty()).map(|(i, p)| (p.contains(PRIMARY), stress_weight(p), i)).collect();
    if idx.len() == 2 && parts[idx[0].2].chars().count() == 1 {
        let i = idx[1].2;
        ps[i] = apply_stress(&ps[i], Some(-0.5));
        return;
    }
    let primaries = idx.iter().filter(|x| x.0).count();
    if idx.len() < 2 || primaries <= idx.len().div_ceil(2) {
        return;
    }
    let mut sorted = idx.clone();
    sorted.sort();
    for &(_, _, i) in sorted.iter().take(idx.len() / 2) {
        ps[i] = apply_stress(&ps[i], Some(-0.5));
    }
}

/// Latin letters with accents → plain letters (café → cafe).
fn fold_accents(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        let r: &str = match c {
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' => "a",
            'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' | 'Ā' => "A",
            'ç' | 'ć' | 'č' => "c",
            'Ç' | 'Ć' | 'Č' => "C",
            'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ę' | 'ě' => "e",
            'È' | 'É' | 'Ê' | 'Ë' | 'Ē' | 'Ę' | 'Ě' => "E",
            'ì' | 'í' | 'î' | 'ï' | 'ī' => "i",
            'Ì' | 'Í' | 'Î' | 'Ï' | 'Ī' => "I",
            'ñ' | 'ń' | 'ň' => "n",
            'Ñ' | 'Ń' | 'Ň' => "N",
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ő' => "o",
            'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' | 'Ø' | 'Ō' | 'Ő' => "O",
            'ù' | 'ú' | 'û' | 'ü' | 'ū' | 'ů' | 'ű' => "u",
            'Ù' | 'Ú' | 'Û' | 'Ü' | 'Ū' | 'Ů' | 'Ű' => "U",
            'ý' | 'ÿ' => "y",
            'Ý' | 'Ÿ' => "Y",
            'ß' => "ss",
            'æ' => "ae",
            'Æ' => "AE",
            'œ' => "oe",
            'Œ' => "OE",
            'š' | 'ś' => "s",
            'Š' | 'Ś' => "S",
            'ž' | 'ź' | 'ż' => "z",
            'Ž' | 'Ź' | 'Ż' => "Z",
            'ł' => "l",
            'Ł' => "L",
            'đ' | 'ð' => "d",
            'þ' => "th",
            _ => {
                out.push(c);
                continue;
            }
        };
        out.push_str(r);
    }
    out
}

/// Final pass: flap and glottal stop in Kokoro's spelling, nothing outside
/// its vocabulary, single spaces.
fn finish(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut space = false;
    for c in s.chars() {
        let c = match c {
            'ɾ' => 'T',
            'ʔ' => 't',
            c => c,
        };
        if c.is_whitespace() {
            space = !out.is_empty();
            continue;
        }
        if !VOCAB.contains(c) {
            continue;
        }
        if space {
            out.push(' ');
            space = false;
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLD: &str = r#"{
        "hello": "həlˈO", "world": "wˈɜɹld", "water": "wˈɔɾəɹ", "apple": "ˈæpᵊl", "the": "ði", "a": "A", "to": "tu", "am": "ˈæm",
        "cat": "kˈæt", "dog": "dˈɔɡ", "walk": "wˈɔk", "want": "wˈɑnt", "need": "nˈid", "play": "plˈA", "bus": "bˈʌs", "sit": "sˈɪt",
        "read": {"ADJ": "ɹˈɛd", "DEFAULT": "ɹˈid", "VBD": "ɹˈɛd", "VBN": "ɹˈɛd", "VBP": "ɹˈɛd"},
        "live": {"DEFAULT": "lˈIv", "VERB": "lˈɪv"},
        "record": {"DEFAULT": "ɹˈɛkəɹd", "VERB": "ɹəkˈɔɹd"},
        "used": {"DEFAULT": "jˈuzd", "VBD": "jˈust"},
        "US": {"DEFAULT": "ˌʌs", "NOUN": null},
        "can": {"DEFAULT": "kæn", "None": "kˈæn"},
        "don't": "dˈOnt", "it's": "ɪts", "I": "ˈI", "i": "ˈI", "'s": "s", "'ll": "əl", "'d": "d",
        "A": "ˈA", "B": "bˈi", "C": "sˈi", "D": "dˈi", "E": "ˈi", "F": "ˈɛf", "G": "ʤˈi", "H": "ˈAʧ", "I": "ˈI", "J": "ʤˈA", "K": "kˈA",
        "L": "ˈɛl", "M": "ˈɛm", "N": "ˈɛn", "O": "ˈO", "P": "pˈi", "Q": "kjˈu", "R": "ˈɑɹ", "S": "ˈɛs", "T": "tˈi", "U": "jˈu", "V": "vˈi",
        "W": "dˈʌbᵊlju", "X": "ˈɛks", "Y": "wˈI", "Z": "zˈi",
        "micro": "mˈIkɹO", "soft": "sˈɔft", "team": "tˈim", "mate": "mˈAt", "follow": "fˈɑlO", "happy": "hˈæpi", "kind": "kˈInd",
        "percent": "pəɹsˈɛnt", "and": "ænd", "plus": "plˈʌs", "at": "æt", "is": "ɪz", "you": "ju", "they": "ðA", "I'm": "ˌIm",
        "one": "wˈʌn", "two": "tˈu", "three": "θɹˈi", "four": "fˈɔɹ", "five": "fˈIv", "zero": "zˈɪɹO", "twenty": "twˈɛnti",
        "hundred": "hˈʌndɹəd", "dollars": "dˈɑləɹz", "point": "pˈYnt", "in": "ɪn", "well": "wˈɛl", "known": "nˈOn", "London": "lˈʌndən",
        "have": "hæv", "my": "mI", "comfortable": "kˈʌmfəɹɾəbᵊl", "gonna": "ɡˈʌnə", "eight": "ˈAt", "seven": "sˈɛvən"
    }"#;
    const SILVER: &str = r#"{"guys": "ɡˈIz"}"#;

    fn g2p() -> G2p {
        G2p::new(Lexicon::from_json(GOLD, SILVER, false).unwrap())
    }

    #[test]
    fn lexicon_words_and_punctuation() {
        let g = g2p();
        assert_eq!(g.phonemize("Hello, world."), "həlˈO, wˈɜɹld.");
        assert_eq!(g.phonemize("water"), "wˈɔTəɹ");
        assert_eq!(g.phonemize("the apple"), "ði ˈæpᵊl");
        assert_eq!(g.phonemize("the cat"), "ðə kˈæt");
        assert_eq!(g.phonemize("HELLO!"), "həlˈO!");
        assert_eq!(g.phonemize("London"), "lˈʌndən");
        assert_eq!(g.phonemize("london"), "lˈʌndən");
        assert_eq!(g.phonemize("\"hello\" (world)"), "“həlˈO” (wˈɜɹld)");
        assert_eq!(g.phonemize("hello - world"), "həlˈO — wˈɜɹld");
        assert_eq!(g.phonemize("hello...  world"), "həlˈO… wˈɜɹld");
        assert_eq!(g.phonemize("guys"), "ɡˈIz");
    }

    #[test]
    fn function_words_follow_context() {
        let g = g2p();
        assert_eq!(g.phonemize("a cat"), "ɐ kˈæt");
        assert_eq!(g.phonemize("to walk"), "tə wˈɔk");
        assert_eq!(g.phonemize("to eight"), "tʊ ˈAt");
        assert_eq!(g.phonemize("I can."), "ˌI kˈæn.");
        assert_eq!(g.phonemize("I can walk"), "ˌI kæn wˈɔk");
        assert_eq!(g.phonemize("used to"), "jˈust tu");
        assert_eq!(g.phonemize("used cat"), "jˈuzd kˈæt");
    }

    #[test]
    fn heteronyms_by_position() {
        let g = g2p();
        assert_eq!(g.phonemize("I read"), "ˌI ɹˈid");
        assert_eq!(g.phonemize("have read"), "hæv ɹˈɛd");
        assert_eq!(g.phonemize("you live"), "ju lˈɪv");
        assert_eq!(g.phonemize("the live"), "ðə lˈIv");
        assert_eq!(g.phonemize("to record"), "tə ɹəkˈɔɹd");
        assert_eq!(g.phonemize("the record"), "ðə ɹˈɛkəɹd");
    }

    #[test]
    fn contractions_and_possessives() {
        let g = g2p();
        assert_eq!(g.phonemize("don't"), "dˈOnt");
        assert_eq!(g.phonemize("don’t"), "dˈOnt");
        assert_eq!(g.phonemize("it's"), "ɪts");
        assert_eq!(g.phonemize("cat's"), "kˈæts");
        assert_eq!(g.phonemize("dog's"), "dˈɔɡz");
        assert_eq!(g.phonemize("bus's"), "bˈʌsᵻz");
        assert_eq!(g.phonemize("cats'"), "kˈæts");
        assert_eq!(g.phonemize("I'm"), "ˌIm");
        // unknown base + known clitic
        let p = g.phonemize("Zorblax'll");
        assert!(p.ends_with("əl"), "{p}");
    }

    #[test]
    fn morphology() {
        let g = g2p();
        assert_eq!(g.phonemize("cats"), "kˈæts");
        assert_eq!(g.phonemize("dogs"), "dˈɔɡz");
        assert_eq!(g.phonemize("walked"), "wˈɔkt");
        assert_eq!(g.phonemize("wanted"), "wˈɑntᵻd");
        assert_eq!(g.phonemize("needed"), "nˈidᵻd");
        assert_eq!(g.phonemize("played"), "plˈAd");
        assert_eq!(g.phonemize("sitting"), "sˈɪTɪŋ");
        assert_eq!(g.phonemize("walking"), "wˈɔkɪŋ");
        // beyond misaki
        assert_eq!(g.phonemize("kindness"), "kˈIndnəs");
        assert_eq!(g.phonemize("happily"), "hˈæpəli");
        assert_eq!(g.phonemize("comfortably"), "kˈʌmfəɹTəbli");
        assert_eq!(g.phonemize("unfollow"), "ˌʌnfˈɑlO");
        assert_eq!(g.phonemize("unfollowed"), "ˌʌnfˈɑlOd");
        assert_eq!(g.phonemize("microsoft"), "mˈIkɹOsˌɔft");
        assert_eq!(g.phonemize("Microsoft"), "mˈIkɹOsˌɔft");
        assert_eq!(g.phonemize("teammates"), "tˈimmˌAts");
        assert_eq!(g.phonemize("well-known"), "wˌɛlnˈOn");
    }

    #[test]
    fn acronyms_are_spelled() {
        let g = g2p();
        assert_eq!(g.phonemize("FBI"), "ˌɛfbˌiˈI");
        assert_eq!(g.phonemize("the US"), "ðə jˌuˈɛs");
        assert_eq!(g.phonemize("U.S."), "jˌuˈɛs.");
        assert_eq!(g.phonemize("mp"), "ˌɛmpˈi");
    }

    #[test]
    fn digits_and_symbols() {
        let g = g2p();
        assert_eq!(g.phonemize("%"), "pəɹsˈɛnt");
        assert_eq!(g.phonemize("cat & dog"), "kˈæt ænd dˈɔɡ");
        let p = g.phonemize("5");
        assert_eq!(p, "fˈIv");
    }

    #[test]
    fn unknown_words_use_letter_to_sound() {
        let g = g2p();
        let p = g.phonemize("Blorptastic");
        assert!(!p.is_empty() && p.contains(PRIMARY), "{p}");
        assert_eq!(g.word("café"), g.word("cafe"));
        assert!(g.word("...").is_none());
        assert!(g.word("").is_none());
        assert_eq!(g.word("hello").as_deref(), Some("həlˈO"));
    }

    #[test]
    fn stress_helpers_match_misaki() {
        assert_eq!(apply_stress("kæt", Some(0.5)), "kˌæt");
        assert_eq!(apply_stress("kæt", Some(2.0)), "kˈæt");
        assert_eq!(apply_stress("kˈæt", Some(-1.0)), "kˌæt");
        assert_eq!(apply_stress("kˈæt", Some(-2.0)), "kæt");
        assert_eq!(apply_stress("kˌæt", Some(1.0)), "kˈæt");
        assert_eq!(apply_stress("st", Some(2.0)), "st");
        assert_eq!(restress("ˈstɹɪŋ"), "stɹˈɪŋ");
    }

    #[test]
    fn output_is_always_in_vocab_and_never_panics() {
        let g = g2p();
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let pool: Vec<char> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'\"-–—…()[]{}%&+@#$€£*/\\éüñçßøåæœ😀中文Ωж\n\t’“”".chars().collect();
        for len in [0usize, 1, 2, 5, 17, 60, 300] {
            for _ in 0..60 {
                let s: String = (0..len)
                    .map(|_| {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        pool[(rng % pool.len() as u64) as usize]
                    })
                    .collect();
                let p = g.phonemize(&s);
                assert!(p.chars().all(|c| VOCAB.contains(c)), "{s:?} → {p:?}");
                for a in super::super::accent::Accent::ALL {
                    let q = super::super::accent::apply(&p, a);
                    assert!(q.chars().all(|c| VOCAB.contains(c)), "{a:?}: {s:?} → {q:?}");
                }
                assert!(!p.contains("  ") && p.trim() == p, "{s:?} → {p:?}");
            }
        }
        let huge = "hello world, zorblax 123. ".repeat(2000);
        let t = std::time::Instant::now();
        assert!(!g.phonemize(&huge).is_empty());
        assert!(t.elapsed().as_secs_f32() < 5.0);
    }

    #[test]
    fn british_morphology() {
        let g = G2p::new(Lexicon::from_json(GOLD, SILVER, true).unwrap());
        assert_eq!(g.phonemize("buses"), "bˈʌsɪz");
        assert_eq!(g.phonemize("wanted"), "wˈɑntɪd");
        assert!(g.british());
    }

    // ---- with the real misaki lexicons (NEKOTONE_KOKORO_DIR) ----

    fn real(british: bool) -> Option<G2p> {
        let dir = std::env::var("NEKOTONE_KOKORO_DIR").ok()?;
        let t = std::time::Instant::now();
        let lex = Lexicon::load(Path::new(&dir), british).expect("load lexicon");
        println!("lexicon ({}) loaded in {:.0} ms", if british { "gb" } else { "us" }, t.elapsed().as_secs_f64() * 1000.0);
        Some(G2p::new(lex))
    }

    #[test]
    #[ignore]
    fn real_lexicon_known_words() {
        let Some(g) = real(false) else { return };
        for (w, want) in [
            ("hello", "həlˈO"),
            ("world", "wˈɜɹld"),
            ("the apple", "ði ˈæpᵊl"),
            ("water", "wˈɔTəɹ"),
            ("Hello, world!", "həlˈO, wˈɜɹld!"),
            ("I can't", "ˌI kˈænt"),
            ("the USA", "ðə jˌuˌɛsˈA"),
        ] {
            assert_eq!(g.phonemize(w), want, "{w}");
        }
    }

    #[test]
    #[ignore]
    fn real_lexicon_sentences() {
        let Some(g) = real(false) else { return };
        let gb = real(true).unwrap();
        let sentences = [
            "Hey guys, I'm gonna grab some coffee and I'll be right back.",
            "The quick brown fox jumps over the lazy dog.",
            "We need to push the objective before the other team respawns.",
            "Did you read the record I sent you yesterday?",
            "I'm not sure that's gonna work, but let's try it anyway.",
            "Can you hear me? My microphone was muted.",
            "The meeting is at three forty five P M on Tuesday.",
            "It costs twenty dollars and fifty cents.",
            "She lives in London, but she works for Microsoft.",
            "Honestly, the dungeon was way harder than I expected.",
            "Wait, wait, wait. Who took the last health potion?",
            "They've been streaming on Twitch for about six hours.",
            "I used to live near the station, now I live downtown.",
            "OK, that's a wrap. Thanks for watching, everybody!",
            "Nekotone turns your speech into a brand new voice.",
            "Could you please repeat the question?",
            "He said \"no\" (twice) — then left…",
            "Unbelievably, the teammates didn't notice the headshot.",
            "Blorptastic zorbling is extraordinarily flibbertigibbety.",
            "The NASA and FBI reports were declassified in the US.",
            "Mister Smith and Doctor Jones met on Baker Street.",
            "Let me know if you're free tomorrow afternoon.",
            "I don't know what you're talking about, mate.",
            "That was absolutely incredible!",
            "Keep your voice down; the baby's asleep.",
            "What's the Wi-Fi password again?",
            "We should've left earlier, the traffic's awful.",
            "Thank you so much for coming, it means a lot.",
            "Press the button, then wait for the green light.",
            "I've never seen anything like it in my entire life.",
            "Our new co-workers are really nice.",
            "The CEO's announcement surprised everyone.",
            "Please don't touch my keyboard.",
            "It's the twenty-first of March, two thousand twenty five.",
            "Grab the loot and let's get out of here!",
            "That's her car, not his.",
            "My brother-in-law lives in Seattle.",
            "Is this thing on? Testing, one, two, three.",
            "Where did you put the charger?",
            "Rewatching old episodes is my guilty pleasure.",
        ];
        let t = std::time::Instant::now();
        let mut total_words = 0;
        for s in sentences {
            let p = g.phonemize(s);
            let q = gb.phonemize(s);
            total_words += s.split_whitespace().count();
            println!("{s}\n  US {p}\n  GB {q}");
            assert!(!p.is_empty() && p.chars().all(|c| VOCAB.contains(c)));
            assert!(q.chars().all(|c| VOCAB.contains(c)));
            assert!(!q.contains('ɹ') || q.contains("ɹ"), "GB output keeps pre-vocalic r only");
        }
        let el = t.elapsed().as_secs_f64();
        println!("{} sentences, {total_words} words, both accents: {:.2} ms per sentence", sentences.len(), el * 1000.0 / (2.0 * sentences.len() as f64));
    }
}
