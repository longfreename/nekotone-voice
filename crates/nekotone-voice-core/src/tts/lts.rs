//! Letter-to-sound rules: the last resort for words that are not in the
//! lexicon and cannot be built from known parts (owner: the G2P package).
//! Context-sensitive rewrite rules in the style of the NRL rules (Elovitz et
//! al., 1976, public domain), written directly into Kokoro's phoneme set.
//!
//! How it works: the word is scanned left to right; at each letter the first
//! rule for that letter whose match string and left/right contexts fit is
//! applied, producing ARPAbet-like codes (with a separate `OH` for the
//! short "o" so British English can say `ɒ`). The codes are mapped to
//! Kokoro's symbols (US: rhotic, diphthongs `A I O W Y`; GB: non-rhotic,
//! `Q` for the "go" vowel, long vowels with `ː`), one primary stress is
//! placed, and unstressed `ʌ`/`ɜɹ` are reduced.
//!
//! Context symbols in the rules: `' '` word boundary, `#` one or more
//! vowels, `.` a voiced consonant, `^` one consonant, `+` e/i/y, `:` zero or
//! more consonants, `%` a suffix (e, er, es, ed, ely, ing), `&` a sibilant,
//! `@` a consonant after which "u" is /u/ not /ju/; letters are literal.
//!
//! Accuracy is measured against held-out words of the misaki US gold
//! lexicon by the ignored test `held_out_accuracy`.

/// One rule: left context, the letters it consumes, right context, output
/// codes (space separated).
type Rule = (&'static str, &'static str, &'static str, &'static str);

// Anything = "", Nothing (boundary) = " ".
static RULES: &[Rule] = &[
    // ---- A ----
    (" ", "a", " ", "AX"),
    ("", "a", " ", "AX"),
    (" ", "are", " ", "AA R"),
    (" ", "ar", "o", "AX R"),
    ("", "ar", "#", "EH R"),
    ("^", "as", "#", "EY S"),
    ("", "a", "wa", "AX"),
    ("", "aw", "", "AO"),
    (" :", "any", "", "EH N IY"),
    ("", "a", "^+#", "EY"),
    ("#:", "ally", "", "AX L IY"),
    (" ", "al", "#", "AX L"),
    ("", "again", "", "AX G EH N"),
    ("#:", "ag", "e", "IH J"),
    ("", "a", "^+:#", "AE"),
    (" :", "a", "^+ ", "EY"),
    ("", "a", "^%", "EY"),
    (" ", "arr", "", "AX R"),
    ("", "arr", "", "AE R"),
    (" :", "ar", " ", "AA R"),
    ("", "ar", " ", "ER"),
    ("", "ar", "", "AA R"),
    ("", "air", "", "EH R"),
    ("", "ai", "", "EY"),
    ("", "ay", "", "EY"),
    ("", "au", "", "AO"),
    ("#:", "al", " ", "EL"),
    ("#:", "als", " ", "EL Z"),
    ("", "alk", "", "AO K"),
    ("", "al", "^", "AO L"),
    (" :", "able", "", "EY B AX L"),
    ("", "able", "", "AX B EL"),
    ("", "ange", "", "EY N J"),
    ("", "ang", "+", "EY N J"),
    ("", "a", "", "AE"),
    // ---- B ----
    (" ", "be", "^#", "B IH"),
    ("", "being", "", "B IY IH NX"),
    (" ", "both", " ", "B OW TH"),
    (" ", "bus", "#", "B IH Z"),
    ("", "buil", "", "B IH L"),
    ("", "bb", "", "B"),
    ("m", "b", " ", ""),
    ("", "b", "", "B"),
    // ---- C ----
    (" ", "ch", "^", "K"),
    ("^e", "ch", "", "K"),
    ("", "chr", "", "K R"),
    ("", "ch", "", "CH"),
    (" s", "ci", "#", "S AY"),
    ("", "ci", "a", "SH"),
    ("", "ci", "o", "SH"),
    ("", "ci", "en", "SH"),
    ("", "cc", "+", "K S"),
    ("", "c", "+", "S"),
    ("", "ck", "", "K"),
    ("", "com", "%", "K AH M"),
    ("", "cc", "", "K"),
    ("", "c", "", "K"),
    // ---- D ----
    ("#:", "ded", " ", "D IX D"),
    (".e", "d", " ", "D"),
    ("#:^e", "d", " ", "T"),
    (" ", "de", "^#", "D IH"),
    (" ", "do", " ", "D UW"),
    (" ", "does", "", "D AH Z"),
    (" ", "doing", "", "D UW IH NX"),
    (" ", "dow", "", "D AW"),
    ("", "du", "a", "J UW"),
    ("", "dd", "", "D"),
    ("", "dg", "e", "J"),
    ("", "d", "", "D"),
    // ---- E ----
    ("#:", "e", " ", ""),
    ("'^:", "e", " ", ""),
    (" :", "e", " ", "IY"),
    ("#", "ed", " ", "D"),
    ("#:", "e", "d ", ""),
    ("", "ev", "er", "EH V"),
    ("", "e", "^%", "IY"),
    ("", "eri", "#", "IY R IY"),
    ("", "eri", "", "EH R IH"),
    ("#:", "er", "#", "ER"),
    ("", "er", "#", "EH R"),
    ("", "er", "", "ER"),
    (" ", "even", "", "IY V EH N"),
    ("#:", "e", "w", ""),
    ("@", "ew", "", "UW"),
    ("", "ew", "", "Y UW"),
    ("", "e", "o", "IY"),
    ("#:&", "es", " ", "IX Z"),
    ("#:", "e", "s ", ""),
    ("#:", "ely", " ", "L IY"),
    ("#:", "ement", "", "M EH N T"),
    ("", "eful", "", "F AX L"),
    ("", "ee", "", "IY"),
    ("", "earn", "", "ER N"),
    (" ", "ear", "^", "ER"),
    ("", "ear", "", "IY R"),
    ("", "ead", "", "EH D"),
    ("#:", "ea", " ", "IY AX"),
    ("", "ea", "su", "EH"),
    ("", "ea", "", "IY"),
    ("", "eigh", "", "EY"),
    ("", "ei", "", "IY"),
    (" ", "eye", "", "AY"),
    ("", "ey", "", "IY"),
    ("", "eu", "", "Y UW"),
    ("", "e", "", "EH"),
    // ---- F ----
    ("", "ful", "", "F AX L"),
    ("", "ff", "", "F"),
    ("", "f", "", "F"),
    // ---- G ----
    ("", "giv", "", "G IH V"),
    (" ", "g", "i^", "G"),
    ("", "ge", "t", "G EH"),
    ("su", "gges", "", "G J EH S"),
    ("", "gg", "", "G"),
    (" b#", "g", "", "G"),
    ("", "g", "+", "J"),
    ("", "great", "", "G R EY T"),
    ("#", "gh", "", ""),
    (" ", "gn", "", "N"),
    ("", "gn", " ", "N"),
    ("", "g", "", "G"),
    // ---- H ----
    ("", "h", "y", "HH"),
    (" ", "hav", "", "HH AE V"),
    (" ", "here", "", "HH IY R"),
    (" ", "hour", "", "AW ER"),
    ("", "how", "", "HH AW"),
    ("", "h", "#", "HH"),
    ("", "h", "", ""),
    // ---- I ----
    ("", "ism", " ", "IH Z AX M"),
    (" ", "in", "", "IH N"),
    (" ", "i", " ", "AY"),
    ("", "in", "d", "AY N"),
    ("", "ier", "", "IY ER"),
    ("#:r", "ied", "", "IY D"),
    ("", "ied", " ", "AY D"),
    ("", "ien", "", "IY EH N"),
    ("", "ie", "t", "AY EH"),
    (" :", "i", "%", "AY"),
    ("", "i", "%", "IY"),
    ("", "ie", "", "IY"),
    ("", "i", "^+:#", "IH"),
    ("", "ir", "#", "AY R"),
    ("", "iz", "%", "AY Z"),
    ("", "is", "%", "AY Z"),
    ("", "i", "d%", "AY"),
    ("+^", "i", "^+", "IH"),
    ("", "i", "t%", "AY"),
    ("#:^", "i", "^+", "IH"),
    ("", "i", "^+", "AY"),
    ("", "ir", "", "ER"),
    ("", "igh", "", "AY"),
    ("", "ild", "", "AY L D"),
    ("", "ign", " ", "AY N"),
    ("", "ign", "^", "AY N"),
    ("", "ign", "%", "AY N"),
    ("", "ique", "", "IY K"),
    ("#:^", "i", " ", "IY"),
    ("", "i", "a", "IY"),
    ("", "i", "o", "IY"),
    ("", "i", "", "IH"),
    // ---- J ----
    ("", "j", "", "J"),
    // ---- K ----
    (" ", "k", "n", ""),
    ("", "k", "", "K"),
    // ---- L ----
    ("#:", "less", " ", "L AX S"),
    ("", "lo", "c#", "L OW"),
    ("l", "l", "", ""),
    ("#:^", "l", "%", "EL"),
    ("", "lead", "", "L IY D"),
    ("", "l", "", "L"),
    // ---- M ----
    ("#:", "ment", " ", "M AX N T"),
    ("", "mov", "", "M UW V"),
    ("", "mm", "", "M"),
    ("", "m", "", "M"),
    // ---- N ----
    ("", "nge", " ", "N J"),
    ("", "nges", " ", "N J IX Z"),
    ("", "nged", " ", "N J D"),
    ("#:", "ness", " ", "N AX S"),
    ("e", "ng", "+", "N J"),
    ("", "ng", "r", "NX G"),
    ("", "ng", "#", "NX G"),
    ("", "ngl", "%", "NX G EL"),
    ("", "ng", "", "NX"),
    ("", "nk", "", "NX K"),
    (" ", "now", " ", "N AW"),
    ("", "nn", "", "N"),
    ("", "n", "", "N"),
    // ---- O ----
    ("", "of", " ", "AX V"),
    ("", "orough", "", "ER OW"),
    ("", "oy", "", "OY"),
    ("", "oi", "", "OY"),
    ("#:", "or", " ", "ER"),
    ("#:", "ors", " ", "ER Z"),
    ("", "or", "", "AO R"),
    (" ", "one", "", "W AH N"),
    ("", "ow", "", "OW"),
    (" ", "over", "", "OW V ER"),
    ("", "ov", "", "AH V"),
    ("", "o", "^%", "OW"),
    ("", "o", "^en", "OW"),
    ("", "o", "^i#", "OW"),
    ("", "ol", "d", "OW L"),
    ("", "ought", "", "AO T"),
    ("", "ough", "", "AH F"),
    (" ", "ou", "", "AW"),
    ("h", "ou", "s#", "AW"),
    ("", "ous", "", "AX S"),
    ("", "our", "", "AW ER"),
    ("", "ould", "", "UH D"),
    ("^", "ou", "^l", "AH"),
    ("", "oup", "", "UW P"),
    ("", "ou", "", "AW"),
    ("", "oing", "", "OW IH NX"),
    ("", "oor", "", "AO R"),
    ("", "ook", "", "UH K"),
    ("", "ood", "", "UH D"),
    ("", "oo", "", "UW"),
    ("", "o", "e", "OW"),
    ("", "o", " ", "OW"),
    ("", "oa", "", "OW"),
    (" ", "only", "", "OW N L IY"),
    (" ", "once", "", "W AH N S"),
    ("", "on't", "", "OW N T"),
    ("c", "o", "n", "OH"),
    ("", "o", "ng", "AO"),
    (" :^", "o", "n", "AH"),
    ("i", "on", "", "AX N"),
    ("#:", "on", " ", "AX N"),
    ("#^", "on", "", "AX N"),
    ("", "o", "st ", "OW"),
    ("", "of", "^", "AO F"),
    ("", "other", "", "AH DH ER"),
    ("", "oss", " ", "AO S"),
    ("#:^", "om", "", "AH M"),
    ("", "o", "", "OH"),
    // ---- P ----
    ("", "psych", "", "S AY K"),
    ("", "ph", "", "F"),
    ("", "peop", "", "P IY P"),
    ("", "pow", "", "P AW"),
    ("", "put", " ", "P UH T"),
    ("", "pp", "", "P"),
    (" ", "ps", "", "S"),
    ("", "p", "", "P"),
    // ---- Q ----
    ("", "quar", "", "K W AO R"),
    ("", "que", " ", "K"),
    ("", "qu", "", "K W"),
    ("", "q", "", "K"),
    // ---- R ----
    (" ", "re", "^#", "R IY"),
    ("", "rh", "", "R"),
    ("", "rr", "", "R"),
    ("", "r", "", "R"),
    // ---- S ----
    ("", "sh", "", "SH"),
    ("#", "sion", "", "ZH AX N"),
    ("", "some", "", "S AH M"),
    ("#", "sur", "#", "ZH ER"),
    ("", "sur", "#", "SH ER"),
    ("#", "su", "#", "ZH UW"),
    ("#", "ssu", "#", "SH UW"),
    ("#", "sed", " ", "Z D"),
    ("#", "s", "#", "Z"),
    ("", "said", "", "S EH D"),
    ("^", "sion", "", "SH AX N"),
    ("", "s", "s", ""),
    (".", "s", " ", "Z"),
    ("#:.e", "s", " ", "Z"),
    ("#:^##", "s", " ", "Z"),
    ("#:^#", "s", " ", "S"),
    ("u", "s", " ", "S"),
    (" :#", "s", " ", "Z"),
    (" ", "sch", "", "S K"),
    ("", "s", "c+", ""),
    ("#", "sm", "", "Z M"),
    ("#", "sn", "'", "Z AX N"),
    ("", "s", "", "S"),
    // ---- T ----
    (" ", "the", " ", "DH AX"),
    ("", "to", " ", "T UW"),
    ("", "that", " ", "DH AE T"),
    (" ", "this", " ", "DH IH S"),
    (" ", "they", "", "DH EY"),
    (" ", "there", "", "DH EH R"),
    ("", "ther", "", "DH ER"),
    ("", "their", "", "DH EH R"),
    (" ", "than", " ", "DH AE N"),
    (" ", "them", " ", "DH EH M"),
    ("", "these", " ", "DH IY Z"),
    (" ", "then", "", "DH EH N"),
    ("", "through", "", "TH R UW"),
    ("", "those", "", "DH OW Z"),
    ("", "though", " ", "DH OW"),
    (" ", "thus", "", "DH AH S"),
    (" ", "ther", "", "TH ER"),
    ("", "th", "", "TH"),
    ("#:", "ted", " ", "T IX D"),
    ("s", "ti", "#n", "CH"),
    ("", "ti", "o", "SH"),
    ("", "ti", "a", "SH"),
    ("", "tien", "", "SH AX N"),
    ("", "tur", "#", "CH ER"),
    ("", "tu", "a", "CH UW"),
    (" ", "two", "", "T UW"),
    ("", "tch", "", "CH"),
    ("", "tt", "", "T"),
    ("", "t", "", "T"),
    // ---- U ----
    (" ", "un", "i", "Y UW N"),
    (" ", "un", "", "AH N"),
    (" ", "upon", "", "AX P AO N"),
    ("@", "ur", "#", "UH R"),
    ("", "ur", "#", "Y UH R"),
    ("", "ur", "", "ER"),
    ("", "u", "^ ", "AH"),
    ("", "u", "^^", "AH"),
    ("", "uy", "", "AY"),
    (" g", "u", "#", ""),
    ("g", "u", "%", ""),
    ("g", "u", "#", "W"),
    ("#n", "u", "", "Y UW"),
    ("@", "u", "", "UW"),
    ("", "u", "", "Y UW"),
    // ---- V ----
    ("", "view", "", "V Y UW"),
    ("", "v", "", "V"),
    // ---- W ----
    (" ", "were", "", "W ER"),
    ("", "wa", "s", "W OH"),
    ("", "wa", "t", "W OH"),
    ("", "where", "", "W EH R"),
    ("", "what", "", "W AH T"),
    ("", "whol", "", "HH OW L"),
    ("", "who", "", "HH UW"),
    ("", "wh", "", "W"),
    ("", "war", "", "W AO R"),
    ("", "wor", "^", "W ER"),
    ("", "wr", "", "R"),
    ("", "w", "", "W"),
    // ---- X ----
    (" ", "x", "", "Z"),
    ("", "x", "", "K S"),
    // ---- Y ----
    ("", "young", "", "Y AH NX"),
    (" ", "you", "", "Y UW"),
    (" ", "yes", "", "Y EH S"),
    (" ", "y", "", "Y"),
    ("#:^", "y", " ", "IY"),
    ("#:^", "y", "i", "IY"),
    (" :", "y", " ", "AY"),
    (" :", "y", "#", "AY"),
    (" :", "y", "^+:#", "IH"),
    (" :", "y", "^#", "AY"),
    ("", "y", "", "IH"),
    // ---- Z ----
    ("", "zz", "", "Z"),
    ("", "z", "", "Z"),
    // ---- apostrophe ----
    ("", "'", "", ""),
];

fn is_vowel(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u')
}
fn is_consonant(c: u8) -> bool {
    c.is_ascii_lowercase() && !is_vowel(c)
}
fn is_voiced(c: u8) -> bool {
    matches!(c, b'b' | b'd' | b'v' | b'g' | b'j' | b'l' | b'm' | b'n' | b'r' | b'w' | b'z')
}
fn is_front(c: u8) -> bool {
    matches!(c, b'e' | b'i' | b'y')
}
fn is_letter(c: u8) -> bool {
    c.is_ascii_lowercase() || c == b'\''
}

/// Right context `pat` starting at `pos` in `w` (outside the word = boundary).
fn right_match(pat: &[u8], w: &[u8], mut pos: usize) -> bool {
    let at = |p: usize| w.get(p).copied().unwrap_or(b' ');
    for &c in pat {
        match c {
            b' ' => {
                if is_letter(at(pos)) {
                    return false;
                }
                pos += 1;
            }
            b'#' => {
                if !is_vowel(at(pos)) {
                    return false;
                }
                while is_vowel(at(pos)) {
                    pos += 1;
                }
            }
            b':' => {
                while is_consonant(at(pos)) {
                    pos += 1;
                }
            }
            b'^' => {
                if !is_consonant(at(pos)) {
                    return false;
                }
                pos += 1;
            }
            b'.' => {
                if !is_voiced(at(pos)) {
                    return false;
                }
                pos += 1;
            }
            b'+' => {
                if !is_front(at(pos)) {
                    return false;
                }
                pos += 1;
            }
            b'%' => {
                // e, er, es, ed, ely, ing
                if at(pos) == b'e' {
                    pos += 1;
                    if at(pos) == b'l' && at(pos + 1) == b'y' {
                        pos += 2;
                    } else if matches!(at(pos), b'r' | b's' | b'd') {
                        pos += 1;
                    }
                } else if at(pos) == b'i' && at(pos + 1) == b'n' && at(pos + 2) == b'g' {
                    pos += 3;
                } else {
                    return false;
                }
            }
            lit => {
                if at(pos) != lit {
                    return false;
                }
                pos += 1;
            }
        }
    }
    true
}

/// Left context `pat` ending just before `end` (read right to left).
fn left_match(pat: &[u8], w: &[u8], end: usize) -> bool {
    let mut pos = end as isize; // the character examined is pos - 1
    let at = |p: isize| if p < 0 { b' ' } else { w.get(p as usize).copied().unwrap_or(b' ') };
    for &c in pat.iter().rev() {
        match c {
            b' ' => {
                if is_letter(at(pos - 1)) {
                    return false;
                }
                pos -= 1;
            }
            b'#' => {
                if !is_vowel(at(pos - 1)) {
                    return false;
                }
                while is_vowel(at(pos - 1)) {
                    pos -= 1;
                }
            }
            b':' => {
                while is_consonant(at(pos - 1)) {
                    pos -= 1;
                }
            }
            b'^' => {
                if !is_consonant(at(pos - 1)) {
                    return false;
                }
                pos -= 1;
            }
            b'.' => {
                if !is_voiced(at(pos - 1)) {
                    return false;
                }
                pos -= 1;
            }
            b'+' => {
                if !is_front(at(pos - 1)) {
                    return false;
                }
                pos -= 1;
            }
            b'&' => {
                let a = at(pos - 1);
                let b = at(pos - 2);
                if a == b'h' && (b == b'c' || b == b's') {
                    pos -= 2;
                } else if matches!(a, b's' | b'c' | b'g' | b'z' | b'x' | b'j') {
                    pos -= 1;
                } else {
                    return false;
                }
            }
            b'@' => {
                let a = at(pos - 1);
                let b = at(pos - 2);
                if a == b'h' && matches!(b, b't' | b'c' | b's') {
                    pos -= 2;
                } else if matches!(a, b't' | b's' | b'r' | b'd' | b'l' | b'z' | b'n' | b'j') {
                    pos -= 1;
                } else {
                    return false;
                }
            }
            lit => {
                if at(pos - 1) != lit {
                    return false;
                }
                pos -= 1;
            }
        }
    }
    true
}

/// Codes for `word` (lower-case ASCII letters and apostrophes) with the
/// letter index each code came from.
fn to_codes(word: &[u8]) -> Vec<(&'static str, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < word.len() {
        let c = word[i];
        let mut applied = false;
        for &(left, m, right, codes) in RULES {
            let mb = m.as_bytes();
            if mb[0] != c || !word[i..].starts_with(mb) {
                continue;
            }
            if !left_match(left.as_bytes(), word, i) || !right_match(right.as_bytes(), word, i + mb.len()) {
                continue;
            }
            for code in codes.split(' ').filter(|s| !s.is_empty()) {
                out.push((code, i));
            }
            i += mb.len();
            applied = true;
            break;
        }
        if !applied {
            i += 1; // a character no rule knows: skip it
        }
    }
    out
}

fn is_vowel_code(c: &str) -> bool {
    matches!(c, "AA" | "OH" | "AE" | "AH" | "AO" | "AW" | "AX" | "AY" | "EH" | "ER" | "EY" | "IH" | "IX" | "EL" | "IY" | "OW" | "OY" | "UH" | "UW")
}

/// Which vowel (index into the nuclei) carries the main stress.
fn stressed_nucleus(word: &str, nuclei: &[usize], letter_of: &[usize]) -> usize {
    let n = nuclei.len();
    if n <= 1 {
        return 0;
    }
    // nucleus index of the last vowel that starts before letter `pos`
    let before = |pos: usize| -> Option<usize> { (0..n).rev().find(|&k| letter_of[nuclei[k]] < pos) };
    let len = word.len();
    // stress on the suffix itself
    for suf in ["eer", "ee", "ese", "ique", "oon", "ette", "esque"] {
        if word.ends_with(suf) && len > suf.len() + 2 {
            return n - 1;
        }
    }
    // stress just before the suffix
    for suf in [
        "tional", "sional", "tion", "sion", "cian", "tian", "ician", "ical", "ically", "ics", "ic", "ity", "ities", "ial", "ially", "ian", "ious",
        "eous", "ual", "ient", "ience", "iency", "ular", "ulous", "itude", "ify", "ogy", "ographer", "ography", "ometer", "ometry", "ologist",
        "ology", "ible", "ibly",
    ] {
        if word.ends_with(suf) && len > suf.len() {
            if let Some(k) = before(len - suf.len()) {
                return k;
            }
        }
    }
    let prefixed = ["be", "de", "re", "con", "com", "ex", "dis", "mis", "en", "em", "pre", "per", "for", "a"]
        .iter()
        .any(|p| word.starts_with(p) && word.len() > p.len() + 2 && (p.len() > 1 || !is_vowel(word.as_bytes()[1])));
    if n == 2 {
        return if prefixed { 1 } else { 0 };
    }
    // three or more: antepenultimate, or the second after an unstressed prefix
    if prefixed && n == 3 {
        return 1;
    }
    n - 3
}

/// Phonemes for a lower-case alphabetic `word`, with one primary stress
/// (`ˈ`) before the stressed vowel. `british` selects non-rhotic vowels.
pub fn letters_to_sounds(word: &str, british: bool) -> String {
    let w: String = word.chars().filter(|c| c.is_ascii_alphabetic() || *c == '\'').collect::<String>().to_ascii_lowercase();
    if w.is_empty() {
        return String::new();
    }
    let codes = to_codes(w.as_bytes());
    if codes.is_empty() {
        return String::new();
    }
    let names: Vec<&str> = codes.iter().map(|c| c.0).collect();
    let letter_of: Vec<usize> = codes.iter().map(|c| c.1).collect();
    let nuclei: Vec<usize> = (0..names.len()).filter(|&i| is_vowel_code(names[i])).collect();
    // syllabic l and the reduced -es/-ed vowel never carry the stress
    let strong: Vec<usize> = nuclei.iter().copied().filter(|&i| !matches!(names[i], "EL" | "IX")).collect();
    let pool = if strong.is_empty() { &nuclei } else { &strong };
    let stressed = if pool.is_empty() { None } else { Some(pool[stressed_nucleus(&w, pool, &letter_of)]) };
    let mut out = String::new();
    let next_is_vowel = |i: usize| names.get(i + 1).map(|c| is_vowel_code(c)).unwrap_or(false);
    let mut i = 0;
    while i < names.len() {
        let c = names[i];
        let st = Some(i) == stressed;
        if st {
            out.push('ˈ');
        }
        // vowel + R not followed by a vowel
        let r_next = names.get(i + 1) == Some(&"R");
        if r_next && !next_is_vowel(i + 1) && is_vowel_code(c) && c != "ER" {
            let s = match (c, british) {
                ("EH" | "AE", false) => "ɛɹ",
                ("EH" | "AE", true) => "ɛː",
                ("AO" | "OW", false) => "ɔɹ",
                ("AO" | "OW", true) => "ɔː",
                ("AA" | "OH", false) => "ɑɹ",
                ("AA" | "OH", true) => "ɑː",
                ("IY" | "IH", false) => "ɪɹ",
                ("IY" | "IH", true) => "ɪə",
                ("UH" | "UW", false) => "ʊɹ",
                ("UH" | "UW", true) => "ʊə",
                ("AY", false) => "Iəɹ",
                ("AY", true) => "Iə",
                ("AW", false) => "Wəɹ",
                ("AW", true) => "Wə",
                (_, false) => "əɹ",
                (_, true) => "ə",
            };
            out.push_str(s);
            i += 2;
            continue;
        }
        // Unstressed full vowels after the first syllable reduce to schwa.
        let first_nucleus = nuclei.first() == Some(&i);
        if !st && !first_nucleus && matches!(c, "AE" | "EH" | "OH" | "AA" | "AO") {
            out.push('ə');
            i += 1;
            continue;
        }
        let s: &str = match c {
            "AA" => {
                if british {
                    "ɑː"
                } else {
                    "ɑ"
                }
            }
            "OH" => {
                if british {
                    "ɒ"
                } else {
                    "ɑ"
                }
            }
            "AE" => {
                if british {
                    "a"
                } else {
                    "æ"
                }
            }
            "AH" => {
                if st {
                    "ʌ"
                } else {
                    "ə"
                }
            }
            "AO" => {
                if british {
                    "ɔː"
                } else {
                    "ɔ"
                }
            }
            "AW" => "W",
            "AX" => "ə",
            "AY" => "I",
            "EH" => "ɛ",
            "ER" => match (st, british) {
                (true, false) => "ɜɹ",
                (true, true) => "ɜː",
                (false, false) => "əɹ",
                (false, true) => {
                    if next_is_vowel(i) {
                        "əɹ"
                    } else {
                        "ə"
                    }
                }
            },
            "EY" => "A",
            "IH" => "ɪ",
            "IX" => "ᵻ",
            "EL" => "ᵊl",
            "IY" => {
                if british && st {
                    "iː"
                } else {
                    "i"
                }
            }
            "OW" => {
                if british {
                    "Q"
                } else {
                    "O"
                }
            }
            "OY" => "Y",
            "UH" => "ʊ",
            "UW" => {
                if british {
                    "uː"
                } else {
                    "u"
                }
            }
            "B" => "b",
            "CH" => "ʧ",
            "D" => "d",
            "DH" => "ð",
            "F" => "f",
            "G" => "ɡ",
            "HH" => "h",
            "J" => "ʤ",
            "K" => "k",
            "L" => "l",
            "M" => "m",
            "N" => "n",
            "NX" => "ŋ",
            "P" => "p",
            "R" => {
                if british && !next_is_vowel(i) {
                    ""
                } else {
                    "ɹ"
                }
            }
            "S" => "s",
            "SH" => "ʃ",
            "T" => "t",
            "TH" => "θ",
            "V" => "v",
            "W" => "w",
            "Y" => "j",
            "Z" => "z",
            "ZH" => "ʒ",
            _ => "",
        };
        out.push_str(s);
        i += 1;
    }
    // Collapse doubled consonants that rules can leave at joins ("kk").
    let mut dedup = String::with_capacity(out.len());
    let mut prev: Option<char> = None;
    for ch in out.chars() {
        if Some(ch) == prev && "bdfɡhklmnpstvwzʃʒθðŋ".contains(ch) {
            continue;
        }
        dedup.push(ch);
        prev = Some(ch);
    }
    dedup
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_patterns() {
        let cases = [
            ("cat", "kˈæt"),
            ("ship", "ʃˈɪp"),
            ("make", "mˈAk"),
            ("bike", "bˈIk"),
            ("home", "hˈOm"),
            ("cute", "kjˈut"),
            ("nation", "nˈAʃən"),
            ("church", "ʧˈɜɹʧ"),
            ("phone", "fˈOn"),
            ("night", "nˈIt"),
            ("king", "kˈɪŋ"),
        ];
        for (w, want) in cases {
            assert_eq!(letters_to_sounds(w, false), want, "{w}");
        }
    }

    #[test]
    fn british_is_non_rhotic() {
        let us = letters_to_sounds("carter", false);
        let gb = letters_to_sounds("carter", true);
        assert!(us.contains('ɹ'), "{us}");
        assert!(!gb.contains('ɹ'), "{gb}");
        assert!(letters_to_sounds("home", true).contains('Q'));
        assert!(letters_to_sounds("hot", true).contains('ɒ'));
    }

    #[test]
    fn stress_follows_suffixes() {
        let s = letters_to_sounds("florbation", false);
        assert!(s.contains("ˈA"), "{s}");
        let s = letters_to_sounds("zorkitee", false);
        assert!(s.ends_with("ˈi"), "{s}");
        for w in ["blorptastic", "unfrobnicate", "zyx", "a", "strengths", "rhythm"] {
            let s = letters_to_sounds(w, false);
            assert!(s.matches('ˈ').count() <= 1, "{w}: {s}");
        }
    }

    #[test]
    fn odd_input_never_panics() {
        for w in ["", "'", "''''", "x", "qqqq", "aeiou", "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", "ÉCOLE", "naïve", "123"] {
            let _ = letters_to_sounds(w, false);
            let _ = letters_to_sounds(w, true);
        }
    }

    fn edit(a: &[char], b: &[char]) -> usize {
        let mut prev: Vec<usize> = (0..=b.len()).collect();
        for (i, ca) in a.iter().enumerate() {
            let mut cur = vec![i + 1; b.len() + 1];
            for (j, cb) in b.iter().enumerate() {
                cur[j + 1] = (prev[j] + (ca != cb) as usize).min(prev[j + 1] + 1).min(cur[j] + 1);
            }
            prev = cur;
        }
        prev[b.len()]
    }

    /// Phoneme error rate (stress marks removed; flap and glottal stop
    /// written `t`) against 2 000 pseudo-random lower-case words of
    /// us_gold.json. Needs NEKOTONE_KOKORO_DIR.
    #[test]
    #[ignore]
    fn held_out_accuracy() {
        let Ok(dir) = std::env::var("NEKOTONE_KOKORO_DIR") else { return };
        let text = std::fs::read_to_string(std::path::Path::new(&dir).join("us_gold.json")).unwrap();
        let map: std::collections::BTreeMap<String, serde_json::Value> = serde_json::from_str(&text).unwrap();
        let words: Vec<(&String, &str)> = map
            .iter()
            .filter(|(k, _)| k.len() >= 3 && k.chars().all(|c| c.is_ascii_lowercase()))
            .filter_map(|(k, v)| v.as_str().map(|s| (k, s)))
            .collect();
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let (mut errs, mut total, mut exact, mut n) = (0usize, 0usize, 0usize, 0usize);
        let norm = |s: &str| -> Vec<char> { s.chars().filter(|c| *c != 'ˈ' && *c != 'ˌ').map(|c| if c == 'ɾ' || c == 'ʔ' { 't' } else { c }).collect() };
        let mut worst = vec![];
        for _ in 0..2000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let (w, gold) = words[(rng % words.len() as u64) as usize];
            let got = norm(&letters_to_sounds(w, false));
            let want = norm(gold);
            let e = edit(&got, &want);
            errs += e;
            total += want.len();
            exact += (e == 0) as usize;
            n += 1;
            if e >= 4 && worst.len() < 30 {
                worst.push(format!("{w}: {} vs {}", got.iter().collect::<String>(), want.iter().collect::<String>()));
            }
        }
        println!("PER {:.1} % ({errs}/{total}), exact {:.1} % of {n}", 100.0 * errs as f64 / total as f64, 100.0 * exact as f64 / n as f64);
        for w in worst {
            println!("  {w}");
        }
    }
}
