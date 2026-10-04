//! Accents for the synthetic voice: rules that rewrite the phonemes of the
//! American or British G2P ([`super::g2p`]) into another English accent,
//! before Kokoro speaks them. Kokoro voices are trained on American and
//! British English, but its phoneme set also holds the sounds these accents
//! need (taps, retroflex stops, glottal stops, long monophthongs), so any
//! voice can speak any accent.
//!
//! The rules are the textbook features that make an accent recognisable
//! (Wells, *Accents of English*), kept to what a phoneme string can say:
//!
//! | accent        | base    | main features |
//! |---------------|---------|---------------|
//! | Australian    | British | FACE [æɪ], PRICE [ɑɪ], GOAT [ɐʊ], MOUTH [æʊ], FLEECE [ɪi], t tapped between vowels |
//! | London        | British | th-fronting (θ→f, ð→v inside words), glottal t, h dropped, wide diphthongs |
//! | Irish         | American| th-stopping (θ→t, ð→d), FACE [eː], GOAT [oː], PRICE [ɐɪ], no tapping |
//! | Scottish      | American| tapped r, FACE [eː], GOAT [oː], FOOT [u], TRAP [a], LOT [ɔ], NURSE [ʌɾ], glottal t |
//! | Southern US   | American| PRICE [aː], pin-pen merger (ɛ→ɪ before n, m), a drawled TRAP [æɛ] |
//! | Indian        | American| retroflex t, d (ʈ ɖ), th-stopping, ʋ for v and w, tapped r, FACE [eː], GOAT [oː], STRUT [ə] |
//!
//! Phoneme conventions (misaki): the stress mark sits right before its
//! vowel; A I W Y O Q are the diphthongs eɪ aɪ aʊ ɔɪ oʊ əʊ; T is the
//! American tapped t; ᵊ a syllabic schwa.

/// An English accent for the synthetic voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Accent {
    /// The voice's own accent (American or British).
    #[default]
    Voice,
    American,
    British,
    Australian,
    London,
    Irish,
    Scottish,
    Southern,
    Indian,
}

impl Accent {
    /// Every accent, in menu order.
    pub const ALL: [Accent; 9] = [
        Accent::Voice,
        Accent::American,
        Accent::British,
        Accent::Australian,
        Accent::London,
        Accent::Irish,
        Accent::Scottish,
        Accent::Southern,
        Accent::Indian,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Accent::Voice => "voice",
            Accent::American => "american",
            Accent::British => "british",
            Accent::Australian => "australian",
            Accent::London => "london",
            Accent::Irish => "irish",
            Accent::Scottish => "scottish",
            Accent::Southern => "southern",
            Accent::Indian => "indian",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Accent::Voice => "The voice's own",
            Accent::American => "American",
            Accent::British => "British (RP)",
            Accent::Australian => "Australian",
            Accent::London => "London",
            Accent::Irish => "Irish",
            Accent::Scottish => "Scottish",
            Accent::Southern => "Southern US",
            Accent::Indian => "Indian English",
        }
    }

    pub fn from_id(id: &str) -> Option<Accent> {
        Accent::ALL.into_iter().find(|a| a.id().eq_ignore_ascii_case(id))
    }

    /// Whether the rules start from the British lexicon (non-rhotic, no
    /// tapping) or the American one; `Voice` keeps the voice's own.
    pub fn british_base(self, voice_is_british: bool) -> bool {
        match self {
            Accent::Voice => voice_is_british,
            Accent::British | Accent::Australian | Accent::London => true,
            Accent::American | Accent::Irish | Accent::Scottish | Accent::Southern | Accent::Indian => false,
        }
    }
}

/// Vowel symbols (including the diphthong letters and what the rules emit).
fn is_vowel(c: char) -> bool {
    "AIOQWYaiueoæɑɒɔəɛɜɪʊʌᵻᵊɐɚ".contains(c)
}

fn is_mark(c: char) -> bool {
    c == 'ˈ' || c == 'ˌ' || c == 'ː'
}

/// The previous phone (skipping stress and length marks) before `i`.
fn prev_phone(p: &[char], i: usize) -> Option<char> {
    p[..i].iter().rev().copied().find(|c| !is_mark(*c))
}

/// The next phone after `i`, and whether a stress mark came first.
fn next_phone(p: &[char], i: usize) -> (Option<char>, bool) {
    let mut stressed = false;
    for &c in &p[i + 1..] {
        if c == 'ˈ' || c == 'ˌ' {
            stressed = true;
        } else if c != 'ː' {
            return (Some(c), stressed);
        }
    }
    (None, stressed)
}

/// Start of a word: nothing before, or a space/punctuation.
fn word_start(p: &[char], i: usize) -> bool {
    i == 0 || !(is_vowel(p[i - 1]) || p[i - 1].is_alphabetic() || "ʤʧθðŋʃʒɾɹ".contains(p[i - 1]) || is_mark(p[i - 1]))
}

/// A t (or tapped T) between a vowel and an unstressed vowel.
fn intervocalic_unstressed(p: &[char], i: usize) -> bool {
    let (next, stressed) = next_phone(p, i);
    matches!(prev_phone(p, i), Some(c) if is_vowel(c)) && matches!(next, Some(c) if is_vowel(c)) && !stressed
}

/// Rewrite phonemes from the base lexicon ([`Accent::british_base`]) into `accent`.
pub fn apply(phonemes: &str, accent: Accent) -> String {
    if matches!(accent, Accent::Voice | Accent::American | Accent::British) {
        return phonemes.to_string();
    }
    let p: Vec<char> = phonemes.chars().collect();
    let mut out = String::with_capacity(phonemes.len() + 16);
    for (i, &c) in p.iter().enumerate() {
        let next = p.get(i + 1).copied();
        let rep: Option<&str> = match accent {
            Accent::Australian => match c {
                'A' => Some("æɪ"),
                'I' => Some("ɑɪ"),
                'Q' | 'O' => Some("ɐʊ"),
                'W' => Some("æʊ"),
                'i' if next == Some('ː') => Some("ɪi"),
                't' if intervocalic_unstressed(&p, i) => Some("T"),
                _ => None,
            },
            Accent::London => match c {
                'A' => Some("æɪ"),
                'I' => Some("ɑɪ"),
                'Q' | 'O' => Some("æʊ"),
                'W' => Some("æː"),
                'θ' => Some("f"),
                'ð' if !word_start(&p, i) => Some("v"),
                't' | 'T' if intervocalic_unstressed(&p, i) || matches!(next, None | Some(' ' | ',' | '.' | '!' | '?' | ';' | ':')) && matches!(prev_phone(&p, i), Some(v) if is_vowel(v)) => Some("ʔ"),
                'h' if word_start(&p, i) => Some(""),
                _ => None,
            },
            Accent::Irish => match c {
                'θ' => Some("t"),
                'ð' => Some("d"),
                'A' => Some("eː"),
                'O' => Some("oː"),
                'I' => Some("ɐɪ"),
                'T' => Some("t"),
                _ => None,
            },
            Accent::Scottish => match c {
                'ɹ' => Some("ɾ"),
                'A' => Some("eː"),
                'O' => Some("oː"),
                'ʊ' => Some("u"),
                'æ' => Some("a"),
                'ɑ' => Some("ɔ"),
                'ɜ' => Some("ʌ"),
                'T' => Some("ʔ"),
                _ => None,
            },
            Accent::Southern => match c {
                'I' => Some("aː"),
                'ɛ' if matches!(next_phone(&p, i).0, Some('n' | 'm')) => Some("ɪ"),
                'æ' if i > 0 && p[i - 1] == 'ˈ' => Some("æɛ"),
                _ => None,
            },
            Accent::Indian => match c {
                't' | 'T' => Some("ʈ"),
                'd' => Some("ɖ"),
                'θ' => Some("t"),
                'ð' => Some("d"),
                'v' | 'w' => Some("ʋ"),
                'ɹ' => Some("ɾ"),
                'A' => Some("eː"),
                'O' => Some("oː"),
                'ʌ' => Some("ə"),
                'ᵊ' => Some("ə"),
                _ => None,
            },
            Accent::Voice | Accent::American | Accent::British => None,
        };
        match rep {
            Some(r) => out.push_str(r),
            None => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kokoro's phoneme vocabulary (as in g2p.rs).
    const VOCAB: &str = ";:,.!?\u{2014}\u{2026}\u{22}()\u{201c}\u{201d} \u{303}ʣʥʦʨᵝ\u{ab67}AIOQSTWYᵊabcdefhijklmnopqrstuvwxyzɑɐɒæβɔɕçɖðʤəɚɛɜɟɡɥɨɪʝɯɰŋɳɲɴøɸθœɹɾɻʁɽʂʃʈʧʊʋʌɣɤχʎʒʔˈˌːʰʲ\u{2193}\u{2192}\u{2197}\u{2198}ᵻ";

    #[test]
    fn every_accent_speaks_only_kokoro_phonemes() {
        let samples = [
            "həlˈO, wˈɜɹld.",
            "wˈɔTəɹ",
            "ði ˈæpᵊl",
            "ðə kˈæt sˈæt ˈɔn ðə mˈæt",
            "ˌI kˈænt θˈɪŋk ˈɛni mˈɔɹ",
            "tˈAk ðə bˈOt tə ðə hˈaʊs, pˈɛn ənd pˈɪn!",
            "bˈʌsɪz wˈɑntɪd",
            "fˈʊt, ɡˈʊd, ˈaɪ, nˈW, bˈQt; ˈiː",
        ];
        for a in Accent::ALL {
            for s in samples {
                let out = apply(s, a);
                assert!(out.chars().all(|c| VOCAB.contains(c)), "{a:?}: {s:?} → {out:?}");
                assert!(!out.is_empty());
            }
        }
    }

    #[test]
    fn accents_change_what_they_should() {
        // American and British bases are left as they are
        assert_eq!(apply("həlˈO", Accent::American), "həlˈO");
        assert_eq!(apply("həlˈQ", Accent::British), "həlˈQ");
        // Australian: FACE, PRICE, GOAT; t tapped between vowels
        assert_eq!(apply("tˈAk mˈI bˈQt", Accent::Australian), "tˈæɪk mˈɑɪ bˈɐʊt");
        assert_eq!(apply("bˈɛtə", Accent::Australian), "bˈɛTə");
        // London: th-fronting, h-dropping, glottal t
        assert_eq!(apply("θˈɪŋk", Accent::London), "fˈɪŋk");
        assert_eq!(apply("hˈQm", Accent::London), "ˈæʊm");
        assert_eq!(apply("bˈɛtə", Accent::London), "bˈɛʔə");
        assert_eq!(apply("ðə", Accent::London), "ðə", "word-initial ð stays");
        assert_eq!(apply("mˈʌðə", Accent::London), "mˈʌvə");
        // Irish: th-stopping, monophthongs, no tapping
        assert_eq!(apply("θˈɪŋk ðə bˈOt", Accent::Irish), "tˈɪŋk də bˈoːt");
        assert_eq!(apply("wˈɔTəɹ", Accent::Irish), "wˈɔtəɹ");
        // Scottish: tapped r, monophthongs, FOOT = GOOSE, TRAP [a]
        assert_eq!(apply("wˈɜɹld", Accent::Scottish), "wˈʌɾld");
        assert_eq!(apply("ɡˈʊd kˈæt", Accent::Scottish), "ɡˈud kˈat");
        assert_eq!(apply("wˈɔTəɹ", Accent::Scottish), "wˈɔʔəɾ");
        // Southern US: PRICE monophthong, pin-pen merger
        assert_eq!(apply("ˈI", Accent::Southern), "ˈaː");
        assert_eq!(apply("pˈɛn", Accent::Southern), "pˈɪn");
        assert_eq!(apply("bˈɛd", Accent::Southern), "bˈɛd", "ɛ only merges before nasals");
        // Indian: retroflex stops, ʋ, tapped r, monophthongs
        assert_eq!(apply("tˈAk dˈɔɹ", Accent::Indian), "ʈˈeːk ɖˈɔɾ");
        assert_eq!(apply("vˈɛɹi wˈɛl", Accent::Indian), "ʋˈɛɾi ʋˈɛl");
    }

    #[test]
    fn bases_and_ids() {
        assert!(Accent::Australian.british_base(false));
        assert!(!Accent::Scottish.british_base(true));
        assert!(Accent::Voice.british_base(true));
        assert!(!Accent::Voice.british_base(false));
        for a in Accent::ALL {
            assert_eq!(Accent::from_id(a.id()), Some(a));
            assert_eq!(serde_json::to_string(&a).unwrap(), format!("\"{}\"", a.id()));
        }
        assert_eq!(Accent::from_id("klingon"), None);
    }
}
