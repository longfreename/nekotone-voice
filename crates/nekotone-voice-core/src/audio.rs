//! Decoding (symphonia), metadata (lofty), resampling (rubato).
//!
//! Owner: the `core-audio` work package. Everything else builds on `Clip`.

use crate::{Error, Result};
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};

use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::units::{Time, TimeBase};

/// File extensions Nekotone opens (lower-case, no dot).
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    // Codecs compiled in (symphonia 0.6, feature "all"): MP1/2/3, AAC, ALAC,
    // FLAC, Vorbis, APE, PCM/ADPCM. Containers: WAV, AIFF, CAF, MP4/M4A/M4B,
    // MKV/WebM, OGG. Not decodable: Opus, WMA, DRM-protected files.
    "mp3", "mp2", "ogg", "oga", "wav", "wave", "flac", "m4a", "m4b", "aac", "mp4", "aiff", "aif", "aifc", "mkv", "webm", "caf", "ape",
];

pub fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| SUPPORTED_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Decoded audio, mono, f32 in -1..1.
#[derive(Debug, Clone, Default)]
pub struct Clip {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Channels in the file before the mix-down (1 or 2 usually).
    pub source_channels: u16,
}

impl Clip {
    pub fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f32 / self.sample_rate as f32
        }
    }
    /// A slice of the clip between two times (clamped).
    pub fn slice(&self, start_secs: f32, end_secs: f32) -> Clip {
        let sr = self.sample_rate as f32;
        let a = ((start_secs * sr) as usize).min(self.samples.len());
        let b = ((end_secs * sr) as usize).clamp(a, self.samples.len());
        Clip { samples: self.samples[a..b].to_vec(), sample_rate: self.sample_rate, source_channels: self.source_channels }
    }
}

/// Tags read from the file (what a player shows).
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Tags {
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub album_artist: Option<String>,
    pub track: Option<u32>,
    pub disc: Option<u32>,
    pub year: Option<u32>,
    pub genre: Option<String>,
    pub comment: Option<String>,
    pub lyrics: Option<String>,
    /// Embedded picture (front cover), raw bytes and MIME type.
    #[serde(skip)]
    pub cover: Option<(Vec<u8>, String)>,
}

/// What `probe` learns without decoding the whole file.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MediaInfo {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub duration_secs: Option<f32>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub codec: Option<String>,
    pub bitrate_kbps: Option<u32>,
    pub tags: Tags,
}

// ---------------------------------------------------------------------------
// symphonia plumbing
// ---------------------------------------------------------------------------

fn decode_err(path: &Path, detail: impl Into<String>) -> Error {
    Error::Decode { path: path.to_path_buf(), detail: detail.into() }
}

fn map_sym(path: &Path, e: SymError) -> Error {
    match e {
        SymError::Unsupported(what) => Error::Unsupported { path: path.to_path_buf(), detail: format!("unsupported {what}") },
        SymError::IoError(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            decode_err(path, "the file ends early; it is probably truncated")
        }
        other => decode_err(path, other.to_string()),
    }
}

/// An opened file: demuxer + decoder + the parameters of the chosen track.
struct Opened {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track_id: u32,
    sample_rate: u32,
    channels: u16,
    time_base: Option<TimeBase>,
    /// Frames in the track as the container states them (may be absent).
    n_frames: Option<u64>,
    /// Duration in seconds from the container, when it says.
    duration_secs: Option<f64>,
    codec: String,
    /// Scratch for interleaved f32 output of one packet.
    scratch: Vec<f32>,
    /// Timestamp (track time base) of the packet whose frames are in `scratch`.
    last_ts: Option<symphonia::core::units::Timestamp>,
}

impl Opened {
    fn open(path: &Path) -> Result<Opened> {
        let file = std::fs::File::open(path)?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(&ext.to_ascii_lowercase());
        }
        let fmt_opts = FormatOptions::default();
        let meta_opts: MetadataOptions = Default::default();
        let format = symphonia::default::get_probe().probe(&hint, mss, fmt_opts, meta_opts).map_err(|e| match e {
            SymError::Unsupported(_) | SymError::DecodeError(_) => Error::Unsupported {
                path: path.to_path_buf(),
                detail: "no known audio container or codec was found in it".into(),
            },
            other => map_sym(path, other),
        })?;
        Self::from_format(path, format)
    }

    fn from_format(path: &Path, format: Box<dyn FormatReader>) -> Result<Opened> {
        let track = format
            .default_track(TrackType::Audio)
            .ok_or_else(|| Error::Unsupported { path: path.to_path_buf(), detail: "it has no audio track".into() })?;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or_else(|| Error::Unsupported { path: path.to_path_buf(), detail: "the audio track has no codec parameters".into() })?
            .clone();
        let registry = symphonia::default::get_codecs();
        let codec = registry
            .get_audio_decoder(params.codec)
            .map(|d| d.codec.info.short_name.to_string())
            .unwrap_or_else(|| params.codec.to_string());
        let decoder = registry.make_audio_decoder(&params, &AudioDecoderOptions::default()).map_err(|e| match e {
            SymError::Unsupported(_) => Error::Unsupported { path: path.to_path_buf(), detail: format!("codec {codec} is not supported") },
            other => map_sym(path, other),
        })?;
        let sample_rate = params.sample_rate.unwrap_or(0);
        let channels = params.channels.as_ref().map(|c| c.count() as u16).unwrap_or(0);
        let time_base = track.time_base;
        let n_frames = track.num_frames;
        let duration_secs = match (time_base, track.duration) {
            (Some(tb), Some(d)) => tb.calc_duration(d).map(|t| t.as_secs_f64()),
            _ => None,
        }
        .or_else(|| n_frames.filter(|_| sample_rate > 0).map(|n| n as f64 / sample_rate as f64));
        Ok(Opened {
            track_id: track.id,
            format,
            decoder,
            sample_rate,
            channels,
            time_base,
            n_frames,
            duration_secs,
            codec,
            scratch: Vec::new(),
            last_ts: None,
        })
    }

    /// Decode the next packet of our track into `scratch` as interleaved f32.
    /// Returns `Ok(Some(frames))`, `Ok(None)` at the end of the stream.
    /// Corrupt packets are skipped (up to a limit); a truncated tail ends the stream.
    fn next_frames(&mut self, path: &Path) -> Result<Option<usize>> {
        let mut bad_packets = 0usize;
        loop {
            let packet = match self.format.next_packet() {
                Ok(Some(p)) => p,
                Ok(None) => return Ok(None),
                Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
                Err(SymError::ResetRequired) => {
                    // The track list changed (chained OGG etc.): rebuild the decoder.
                    let track = self
                        .format
                        .default_track(TrackType::Audio)
                        .ok_or_else(|| decode_err(path, "the stream changed and no audio track remained"))?;
                    let params = track
                        .codec_params
                        .as_ref()
                        .and_then(|p| p.audio())
                        .ok_or_else(|| decode_err(path, "the stream changed to an unknown codec"))?
                        .clone();
                    self.track_id = track.id;
                    self.decoder = symphonia::default::get_codecs()
                        .make_audio_decoder(&params, &AudioDecoderOptions::default())
                        .map_err(|e| map_sym(path, e))?;
                    continue;
                }
                Err(SymError::DecodeError(_)) => {
                    bad_packets += 1;
                    if bad_packets > 64 {
                        return Err(decode_err(path, "too many damaged packets in a row"));
                    }
                    continue;
                }
                Err(e) => return Err(map_sym(path, e)),
            };
            if packet.track_id != self.track_id {
                continue;
            }
            match self.decoder.decode(&packet) {
                Ok(buf) => {
                    let frames = buf.frames();
                    if frames == 0 {
                        continue;
                    }
                    let spec = buf.spec();
                    if self.sample_rate == 0 {
                        self.sample_rate = spec.rate();
                    }
                    let ch = spec.channels().count() as u16;
                    if self.channels == 0 || self.channels != ch {
                        self.channels = ch;
                    }
                    copy_interleaved(&buf, &mut self.scratch);
                    self.last_ts = Some(packet.pts);
                    return Ok(Some(frames));
                }
                Err(SymError::DecodeError(_)) => {
                    bad_packets += 1;
                    if bad_packets > 64 {
                        return Err(decode_err(path, "too many damaged packets in a row"));
                    }
                    continue;
                }
                Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
                Err(SymError::IoError(_)) => continue,
                Err(e) => return Err(map_sym(path, e)),
            }
        }
    }
}

fn copy_interleaved(buf: &GenericAudioBufferRef<'_>, out: &mut Vec<f32>) {
    out.resize(buf.samples_interleaved(), 0.0);
    buf.copy_to_slice_interleaved(&mut out[..]);
}

/// Decode a whole file, calling `sink(interleaved, channels)` per packet.
fn decode_all(path: &Path, mut sink: impl FnMut(&[f32], u16)) -> Result<Opened> {
    let mut o = Opened::open(path)?;
    let mut any = false;
    while let Some(frames) = o.next_frames(path)? {
        let ch = o.channels.max(1) as usize;
        let n = frames * ch;
        if o.scratch.len() >= n {
            sink(&o.scratch[..n], ch as u16);
            any = true;
        }
    }
    if !any && o.sample_rate == 0 {
        return Err(decode_err(path, "no audio could be decoded from it"));
    }
    Ok(o)
}

/// Read metadata and stream parameters. Cheap; used while scanning.
pub fn probe(path: &Path) -> Result<MediaInfo> {
    if !path.is_file() {
        return Err(Error::NotFound(path.to_path_buf()));
    }
    let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let mut info = MediaInfo { path: path.to_path_buf(), size_bytes, ..Default::default() };

    // Tags and (fallback) properties from lofty. A lofty failure is not fatal:
    // symphonia decides whether the file is readable.
    let mut lofty_duration = None;
    let mut lofty_bitrate = None;
    if let Ok(tf) = lofty::probe::Probe::open(path)
        .map(|p| p.options(lofty::config::ParseOptions::new().read_cover_art(true)))
        .and_then(|p| p.guess_file_type().map_err(Into::into))
        .and_then(|p| p.read())
    {
        use lofty::file::{AudioFile, TaggedFileExt};
        use lofty::tag::{Accessor, ItemKey};
        let props = tf.properties();
        let d = props.duration().as_secs_f32();
        if d > 0.0 {
            lofty_duration = Some(d);
        }
        lofty_bitrate = props.audio_bitrate().or(props.overall_bitrate());
        if let Some(tag) = tf.primary_tag().or_else(|| tf.first_tag()) {
            let s = |c: Option<std::borrow::Cow<'_, str>>| c.map(|c| c.trim().to_string()).filter(|s| !s.is_empty());
            let t = &mut info.tags;
            t.title = s(tag.title());
            t.artist = s(tag.artist());
            t.album = s(tag.album());
            t.genre = s(tag.genre());
            t.comment = s(tag.comment());
            t.track = tag.track();
            t.disc = tag.disk();
            t.year = tag.date().map(|d| d.year as u32).or_else(|| {
                // ID3v2.4 keeps the year inside TDRC; other formats use Year or the release date.
                [ItemKey::Year, ItemKey::RecordingDate, ItemKey::OriginalReleaseDate, ItemKey::ReleaseDate]
                    .iter()
                    .filter_map(|k| tag.get_string(*k))
                    .find_map(|y| y.trim().chars().take(4).collect::<String>().parse::<u32>().ok().filter(|y| (1000..=9999).contains(y)))
            });
            t.album_artist = tag.get_string(ItemKey::AlbumArtist).map(|s| s.to_string()).filter(|s| !s.is_empty());
            t.lyrics = tag
                .get_string(ItemKey::UnsyncLyrics)
                .or_else(|| tag.get_string(ItemKey::Lyrics))
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty());
            let pics = tag.pictures();
            let pic = pics
                .iter()
                .find(|p| p.pic_type() == lofty::picture::PictureType::CoverFront)
                .or_else(|| pics.first());
            if let Some(p) = pic {
                let mime = p.mime_type().map(|m| m.as_str().to_string()).unwrap_or_else(|| "application/octet-stream".into());
                t.cover = Some((p.data().to_vec(), mime));
            }
        }
    }

    // Stream parameters from symphonia (authoritative for what we can decode).
    let o = Opened::open(path)?;
    info.codec = Some(o.codec.clone());
    if o.sample_rate > 0 {
        info.sample_rate = Some(o.sample_rate);
    }
    if o.channels > 0 {
        info.channels = Some(o.channels);
    }
    let mut duration = o.duration_secs.map(|d| d as f32).filter(|d| *d > 0.0).or(lofty_duration);
    if duration.is_none() || o.sample_rate == 0 {
        // Nothing states the length: count the frames by decoding.
        let mut frames = 0u64;
        let o2 = decode_all(path, |buf, ch| frames += (buf.len() / ch.max(1) as usize) as u64)?;
        if o2.sample_rate > 0 {
            duration = Some(frames as f32 / o2.sample_rate as f32);
            info.sample_rate = Some(o2.sample_rate);
            info.channels = Some(o2.channels);
        }
    }
    info.duration_secs = duration;
    info.bitrate_kbps = lofty_bitrate.filter(|b| *b > 0).or_else(|| {
        duration.filter(|d| *d > 0.05).map(|d| ((size_bytes as f64 * 8.0) / (d as f64 * 1000.0)).round() as u32)
    });
    Ok(info)
}

/// Decode the whole file to mono f32 at its native sample rate.
pub fn decode(path: &Path) -> Result<Clip> {
    if !path.is_file() {
        return Err(Error::NotFound(path.to_path_buf()));
    }
    let mut samples: Vec<f32> = Vec::new();
    let o = decode_all(path, |buf, ch| {
        let ch = ch.max(1) as usize;
        match ch {
            1 => samples.extend_from_slice(buf),
            2 => samples.extend(buf.chunks_exact(2).map(|f| 0.5 * (f[0] + f[1]))),
            n => {
                let g = 1.0 / n as f32;
                samples.extend(buf.chunks_exact(n).map(|f| f.iter().sum::<f32>() * g));
            }
        }
    })?;
    if let Some(n) = o.n_frames {
        // Trust the container over a few stray padding frames.
        let n = n as usize;
        if samples.len() > n && samples.len() - n < 4096 {
            samples.truncate(n);
        }
    }
    Ok(Clip { samples, sample_rate: o.sample_rate, source_channels: o.channels.max(1) })
}

/// Decode to interleaved stereo f32 at the native rate (for playback and
/// for writing files). Mono sources are duplicated to both channels.
pub fn decode_stereo(path: &Path) -> Result<(Vec<f32>, u32)> {
    if !path.is_file() {
        return Err(Error::NotFound(path.to_path_buf()));
    }
    let mut out: Vec<f32> = Vec::new();
    let o = decode_all(path, |buf, ch| to_stereo(buf, ch, &mut out))?;
    Ok((out, o.sample_rate))
}

/// Append `buf` (interleaved, `ch` channels) to `out` as interleaved stereo.
fn to_stereo(buf: &[f32], ch: u16, out: &mut Vec<f32>) {
    match ch.max(1) as usize {
        1 => {
            out.reserve(buf.len() * 2);
            for &s in buf {
                out.push(s);
                out.push(s);
            }
        }
        2 => out.extend_from_slice(buf),
        n => {
            // Fold surround into a stereo pair: even channels left, odd right,
            // the centre channel (index 2 in WAVE order) into both.
            out.reserve(buf.len() / n * 2);
            for f in buf.chunks_exact(n) {
                let (mut l, mut r, mut nl, mut nr) = (0.0f32, 0.0f32, 0u32, 0u32);
                for (i, &s) in f.iter().enumerate() {
                    if i == 2 {
                        l += s;
                        r += s;
                        nl += 1;
                        nr += 1;
                    } else if i % 2 == 0 {
                        l += s;
                        nl += 1;
                    } else {
                        r += s;
                        nr += 1;
                    }
                }
                out.push(l / nl.max(1) as f32);
                out.push(r / nr.max(1) as f32);
            }
        }
    }
}

/// High-quality resample (rubato, sinc) to `target_rate`.
pub fn resample(clip: &Clip, target_rate: u32) -> Result<Clip> {
    if clip.sample_rate == target_rate {
        return Ok(clip.clone());
    }
    if clip.sample_rate == 0 || target_rate == 0 {
        return Err(Error::Other(anyhow::anyhow!("cannot resample: a sample rate of 0 was given")));
    }
    if clip.samples.is_empty() {
        return Ok(Clip { samples: vec![], sample_rate: target_rate, source_channels: clip.source_channels });
    }
    let samples = resample_mono(&clip.samples, clip.sample_rate, target_rate)?;
    Ok(Clip { samples, sample_rate: target_rate, source_channels: clip.source_channels })
}

/// Band-limited (Kaiser-windowed sinc) resampling of a mono buffer with
/// zero delay: sample `n` of the output lines up exactly with time
/// `n / to` of the input. (The earlier rubato path led the input by a
/// sample and drifted fractionally, which broke `mix − vocals` in stems.)
pub(crate) fn resample_mono(input: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == 0 || to == 0 {
        return Err(Error::Other(anyhow::anyhow!("cannot resample: a sample rate of 0 was given")));
    }
    Ok(crate::process::resample_sinc(input, from, to))
}

/// Write a mono or interleaved-stereo f32 buffer as 16-bit PCM WAV,
/// creating the folder when it does not exist yet.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32, channels: u16) -> Result<()> {
    if !(1..=2).contains(&channels) {
        return Err(Error::Other(anyhow::anyhow!("write_wav: {channels} channels requested; only mono or stereo can be written")));
    }
    if sample_rate == 0 {
        return Err(Error::Other(anyhow::anyhow!("write_wav: the sample rate must be greater than 0")));
    }
    let frames = samples.len() / channels as usize;
    let data_len = (frames * channels as usize * 2) as u32;
    let block_align = channels * 2;
    let byte_rate = sample_rate * block_align as u32;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    w.write_all(b"RIFF")?;
    w.write_all(&(36 + data_len).to_le_bytes())?;
    w.write_all(b"WAVE")?;
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?; // PCM
    w.write_all(&channels.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&16u16.to_le_bytes())?;
    w.write_all(b"data")?;
    w.write_all(&data_len.to_le_bytes())?;
    let mut buf = Vec::with_capacity(data_len as usize);
    for &s in &samples[..frames * channels as usize] {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    w.write_all(&buf)?;
    w.flush()?;
    Ok(())
}

/// Streaming decoder for the player: pulls interleaved stereo frames on
/// demand and can seek.
pub trait Source: Send {
    fn sample_rate(&self) -> u32;
    fn channels(&self) -> u16;
    fn duration_secs(&self) -> Option<f32>;
    /// Fill `out` (interleaved), return frames written; 0 at the end.
    fn read(&mut self, out: &mut [f32]) -> Result<usize>;
    fn seek(&mut self, secs: f32) -> Result<()>;
}

struct FileSource {
    path: PathBuf,
    opened: Opened,
    /// Interleaved stereo frames decoded but not yet handed out.
    pending: Vec<f32>,
    pending_pos: usize,
    /// Frames to drop after a seek (the packet started before the target).
    skip_frames: usize,
    /// After an accurate seek: drop everything before this time, judged by
    /// each decoded packet's own timestamp. (Counting from the demuxer's
    /// landing page is wrong for Vorbis/Opus, whose first packet after a
    /// reset primes the decoder and yields no frames.)
    skip_until_secs: Option<f64>,
    duration: Option<f32>,
    eof: bool,
}

impl FileSource {
    fn refill(&mut self) -> Result<bool> {
        self.pending.clear();
        self.pending_pos = 0;
        loop {
            match self.opened.next_frames(&self.path)? {
                None => {
                    self.eof = true;
                    return Ok(false);
                }
                Some(frames) => {
                    let ch = self.opened.channels.max(1) as usize;
                    let mut start = 0usize;
                    if let Some(req) = self.skip_until_secs {
                        match (self.opened.time_base, self.opened.last_ts) {
                            (Some(tb), Some(ts)) => {
                                let sr = self.opened.sample_rate.max(1) as f64;
                                let pts = tb.calc_time_saturating(ts).as_secs_f64();
                                if pts + frames as f64 / sr <= req + 1e-9 {
                                    continue;
                                }
                                start = ((req - pts).max(0.0) * sr).round() as usize;
                            }
                            _ => start = 0,
                        }
                        self.skip_until_secs = None;
                    }
                    if self.skip_frames > 0 {
                        let s = self.skip_frames.min(frames);
                        self.skip_frames -= s;
                        start = s;
                    }
                    if start >= frames {
                        continue;
                    }
                    let n = frames * ch;
                    if self.opened.scratch.len() < n {
                        continue;
                    }
                    let buf = std::mem::take(&mut self.opened.scratch);
                    to_stereo(&buf[start * ch..n], ch as u16, &mut self.pending);
                    self.opened.scratch = buf;
                    return Ok(true);
                }
            }
        }
    }
}

impl Source for FileSource {
    fn sample_rate(&self) -> u32 {
        self.opened.sample_rate
    }
    fn channels(&self) -> u16 {
        2
    }
    fn duration_secs(&self) -> Option<f32> {
        self.duration
    }
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let want = out.len() / 2 * 2;
        let mut written = 0usize;
        while written < want {
            if self.pending_pos >= self.pending.len() && (self.eof || !self.refill()?) {
                break;
            }
            let avail = self.pending.len() - self.pending_pos;
            let n = avail.min(want - written);
            out[written..written + n].copy_from_slice(&self.pending[self.pending_pos..self.pending_pos + n]);
            self.pending_pos += n;
            written += n;
        }
        Ok(written / 2)
    }
    fn seek(&mut self, secs: f32) -> Result<()> {
        let secs = secs.max(0.0) as f64;
        let time = Time::try_from_secs_f64(secs).ok_or_else(|| decode_err(&self.path, "seek time out of range"))?;
        let to = SeekTo::Time { time, track_id: Some(self.opened.track_id) };
        self.pending.clear();
        self.pending_pos = 0;
        self.skip_frames = 0;
        self.skip_until_secs = None;
        self.eof = false;
        if let Some(d) = self.duration {
            if secs >= d as f64 {
                self.eof = true;
                return Ok(());
            }
        }
        match self.opened.format.seek(SeekMode::Accurate, to) {
            Ok(seeked) => {
                self.opened.decoder.reset();
                // The demuxer lands on a packet at or before the target; the
                // frames in between are dropped as they are decoded, by
                // packet timestamp (see `skip_until_secs`).
                let _ = seeked.actual_ts;
                self.skip_until_secs = self.opened.time_base.map(|tb| tb.calc_time_saturating(seeked.required_ts).as_secs_f64()).or(Some(secs));
                Ok(())
            }
            Err(SymError::SeekError(_)) | Err(SymError::IoError(_)) => {
                // Unseekable stream or a seek the demuxer cannot do: reopen
                // from the start and skip forward by decoding.
                let mut reopened = Opened::open(&self.path)?;
                std::mem::swap(&mut self.opened, &mut reopened);
                self.skip_frames = (secs * self.opened.sample_rate as f64).round() as usize;
                Ok(())
            }
            Err(e) => Err(map_sym(&self.path, e)),
        }
    }
}

/// Open a file as a streaming [`Source`].
pub fn open_source(path: &Path) -> Result<Box<dyn Source>> {
    if !path.is_file() {
        return Err(Error::NotFound(path.to_path_buf()));
    }
    let opened = Opened::open(path)?;
    let duration = opened.duration_secs.map(|d| d as f32).filter(|d| *d > 0.0).or_else(|| {
        // No length in the header (some WAVs, streamed OGGs): ask lofty.
        lofty::read_from_path(path).ok().and_then(|tf| {
            use lofty::file::AudioFile;
            let d = tf.properties().duration().as_secs_f32();
            (d > 0.0).then_some(d)
        })
    });
    Ok(Box::new(FileSource {
        path: path.to_path_buf(),
        opened,
        pending: Vec::new(),
        pending_pos: 0,
        skip_frames: 0,
        skip_until_secs: None,
        duration,
        eof: false,
    }))
}

/// Content hash of the file bytes (blake3), for duplicate detection and
/// change tracking in the index.
pub fn file_hash(path: &Path) -> Result<String> {
    let mut hasher = blake3::Hasher::new();
    let mut f = std::fs::File::open(path)?;
    std::io::copy(&mut f, &mut hasher)?;
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(sr: u32, hz: f32, secs: f32) -> Vec<f32> {
        (0..(sr as f32 * secs) as usize).map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / sr as f32).sin() * 0.5).collect()
    }

    #[test]
    fn write_wav_creates_missing_folder() {
        // The MIDI preview writes into %TEMP%/nekotone-previews, which may not exist yet.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-yet").join("deeper").join("preview.wav");
        write_wav(&path, &[0.0, 0.5, -0.5, 0.0], 22_050, 1).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 44 + 8);
    }

    #[test]
    fn wav_roundtrip_mono_and_stereo() {
        let dir = tempfile::tempdir().unwrap();
        let mono = dir.path().join("m.wav");
        let s = sine(22_050, 440.0, 0.5);
        write_wav(&mono, &s, 22_050, 1).unwrap();
        let c = decode(&mono).unwrap();
        assert_eq!(c.sample_rate, 22_050);
        assert_eq!(c.source_channels, 1);
        assert_eq!(c.samples.len(), s.len());
        let err: f32 = c.samples.iter().zip(&s).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 1.0 / 32000.0, "max error {err}");

        let stereo = dir.path().join("s.wav");
        let mut inter = Vec::new();
        for (i, &v) in s.iter().enumerate() {
            inter.push(v);
            inter.push(if i % 2 == 0 { 0.25 } else { -0.25 });
        }
        write_wav(&stereo, &inter, 48_000, 2).unwrap();
        let (st, sr) = decode_stereo(&stereo).unwrap();
        assert_eq!(sr, 48_000);
        assert_eq!(st.len(), inter.len());
        let c = decode(&stereo).unwrap();
        assert_eq!(c.source_channels, 2);
        assert_eq!(c.samples.len(), s.len());
        let info = probe(&stereo).unwrap();
        assert_eq!(info.channels, Some(2));
        assert_eq!(info.sample_rate, Some(48_000));
        assert!((info.duration_secs.unwrap() - s.len() as f32 / 48_000.0).abs() < 0.01);
        assert!(info.codec.as_deref().unwrap_or("").starts_with("pcm"), "codec {:?}", info.codec);
        assert!(info.bitrate_kbps.unwrap() > 1000);
    }

    #[test]
    fn truncated_wav_is_a_decode_error_or_short_read() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.wav");
        write_wav(&p, &sine(16_000, 300.0, 1.0), 16_000, 1).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        // Chop half the data: header still claims the full length.
        std::fs::write(&p, &bytes[..bytes.len() / 2]).unwrap();
        let r = decode(&p);
        match r {
            Ok(c) => assert!(c.samples.len() < 16_000 && c.samples.len() > 1000),
            Err(Error::Decode { .. }) => {}
            Err(e) => panic!("unexpected error {e}"),
        }
        // Garbage is a readable error, never a panic.
        let g = dir.path().join("g.ogg");
        std::fs::write(&g, vec![0x55u8; 5000]).unwrap();
        match decode(&g) {
            Err(Error::Decode { .. }) | Err(Error::Unsupported { .. }) => {}
            other => panic!("garbage decoded? {:?}", other.map(|c| c.samples.len())),
        }
        // Header only.
        let h = dir.path().join("h.wav");
        std::fs::write(&h, &bytes[..44]).unwrap();
        match decode(&h) {
            Ok(c) => assert!(c.samples.is_empty()),
            Err(Error::Decode { .. }) => {}
            Err(e) => panic!("unexpected error {e}"),
        }
        assert!(matches!(decode(Path::new("does/not/exist.wav")), Err(Error::NotFound(_))));
    }

    #[test]
    fn resample_keeps_length_and_tone() {
        let s = sine(44_100, 1000.0, 1.0);
        let clip = Clip { samples: s, sample_rate: 44_100, source_channels: 1 };
        let r = resample(&clip, 16_000).unwrap();
        assert_eq!(r.sample_rate, 16_000);
        assert!((r.samples.len() as i64 - 16_000).abs() <= 2, "len {}", r.samples.len());
        // Amplitude preserved (skip the filter edges).
        let peak = r.samples[500..15_500].iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        assert!((peak - 0.5).abs() < 0.02, "peak {peak}");
        // Zero crossings per second ~ 2000 for a 1 kHz tone.
        let zc = r.samples.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        assert!((zc as i64 - 2000).abs() < 30, "zero crossings {zc}");
        // No click at the end: the last samples are continuous.
        let tail = &r.samples[r.samples.len() - 8..];
        for w in tail.windows(2) {
            assert!((w[0] - w[1]).abs() < 0.25, "tail jump {:?}", tail);
        }
        // Upsampling too.
        let up = resample(&clip, 48_000).unwrap();
        assert!((up.samples.len() as i64 - 48_000).abs() <= 2);
        let short = Clip { samples: vec![0.1; 50], sample_rate: 44_100, source_channels: 1 };
        let rs = resample(&short, 16_000).unwrap();
        assert!(rs.samples.len() <= 20);
    }

    #[test]
    fn source_reads_and_seeks() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("src.wav");
        let sr = 8_000u32;
        // A ramp so positions are recognisable: sample i = i / n.
        let n = sr as usize * 2;
        let ramp: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        write_wav(&p, &ramp, sr, 1).unwrap();
        let mut src = open_source(&p).unwrap();
        assert_eq!(src.channels(), 2);
        assert_eq!(src.sample_rate(), sr);
        assert!((src.duration_secs().unwrap() - 2.0).abs() < 0.01);
        let mut buf = vec![0.0f32; 512];
        let got = src.read(&mut buf).unwrap();
        assert_eq!(got, 256);
        assert!((buf[0] - 0.0).abs() < 1e-3);
        assert!((buf[1] - buf[0]).abs() < 1e-6, "stereo duplicate");
        // Seek to 1.5 s: the next frame should be ~0.75.
        src.seek(1.5).unwrap();
        let got = src.read(&mut buf).unwrap();
        assert!(got > 0);
        assert!((buf[0] - 0.75).abs() < 0.002, "after seek got {}", buf[0]);
        // Read to the end.
        let mut total = got;
        loop {
            let g = src.read(&mut buf).unwrap();
            if g == 0 {
                break;
            }
            total += g;
        }
        assert!((total as i64 - (n / 4) as i64).abs() <= 2, "frames after seek {total}");
        // Seek back to the start works after EOF.
        src.seek(0.0).unwrap();
        let got = src.read(&mut buf).unwrap();
        assert_eq!(got, 256);
        assert!(buf[0].abs() < 1e-3);
        // Seek past the end: reads 0.
        src.seek(10.0).unwrap();
        assert_eq!(src.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn probe_reads_tags_and_cover() {
        use lofty::config::WriteOptions;
        use lofty::picture::{MimeType, Picture, PictureType};
        use lofty::tag::{Accessor, ItemKey, Tag, TagExt, TagType};
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("tagged.wav");
        write_wav(&p, &sine(16_000, 440.0, 0.3), 16_000, 1).unwrap();
        let mut tag = Tag::new(TagType::Id3v2);
        tag.set_title("Neko Song".into());
        tag.set_artist("The Cats".into());
        tag.set_album("Purr".into());
        tag.set_genre("Game".into());
        tag.set_track(3);
        tag.insert_text(ItemKey::RecordingDate, "2024".into());
        tag.insert_text(ItemKey::AlbumArtist, "Various Cats".into());
        tag.insert_text(ItemKey::UnsyncLyrics, "meow meow".into());
        let png = vec![0x89, b'P', b'N', b'G', 13, 10, 26, 10, 1, 2, 3, 4];
        tag.push_picture(Picture::unchecked(png.clone()).pic_type(PictureType::CoverFront).mime_type(MimeType::Png).build());
        tag.save_to_path(&p, WriteOptions::default()).unwrap();

        let info = probe(&p).unwrap();
        assert_eq!(info.tags.title.as_deref(), Some("Neko Song"));
        assert_eq!(info.tags.artist.as_deref(), Some("The Cats"));
        assert_eq!(info.tags.album.as_deref(), Some("Purr"));
        assert_eq!(info.tags.album_artist.as_deref(), Some("Various Cats"));
        assert_eq!(info.tags.genre.as_deref(), Some("Game"));
        assert_eq!(info.tags.lyrics.as_deref(), Some("meow meow"));
        assert_eq!(info.tags.track, Some(3));
        assert_eq!(info.tags.year, Some(2024));
        let (data, mime) = info.tags.cover.as_ref().expect("cover");
        assert_eq!(data, &png);
        assert_eq!(mime, "image/png");
        assert_eq!(info.sample_rate, Some(16_000));
        assert!((info.duration_secs.unwrap() - 0.3).abs() < 0.01);
        // The tag chunk does not disturb decoding.
        let c = decode(&p).unwrap();
        assert_eq!(c.samples.len(), 4800);
        // An untagged file has empty tags, not an error.
        let plain = dir.path().join("plain.wav");
        write_wav(&plain, &sine(8_000, 200.0, 0.2), 8_000, 1).unwrap();
        assert_eq!(probe(&plain).unwrap().tags, Tags::default());
    }

    /// Smoke test over the real DDO client sounds when that folder exists
    /// (read-only): probe and decode agree, nothing panics, and a seek on
    /// the streaming source lands where the decoded clip says it should.
    #[test]
    fn real_ddo_files_probe_decode_and_seek() {
        let dir = Path::new(r"C:\Users\Adam Bennett\Development\artifacts\client-assets\audio\client_sound");
        if !dir.is_dir() {
            eprintln!("skipped: {} is not present", dir.display());
            return;
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).filter(|p| is_supported(p)).collect();
        files.sort();
        let step = (files.len() / 40).max(1);
        let (mut n, mut seeks, mut worst_lag) = (0usize, 0usize, 0i64);
        for p in files.iter().step_by(step).take(40) {
            let info = probe(p).unwrap_or_else(|e| panic!("{e}"));
            let clip = decode(p).unwrap_or_else(|e| panic!("{e}"));
            assert!(clip.sample_rate > 0 && !clip.samples.is_empty(), "{}", p.display());
            let d = info.duration_secs.expect("duration from probe");
            assert!((d - clip.duration_secs()).abs() < 0.1 + d * 0.02, "{}: probe {d}s vs decode {}s", p.display(), clip.duration_secs());
            assert!(info.codec.is_some() && info.bitrate_kbps.is_some());
            n += 1;
            if clip.duration_secs() < 2.5 {
                continue;
            }
            // Seek to 1 s and compare the streamed mono mix with the decoded clip.
            let mut src = open_source(p).unwrap();
            assert!((src.duration_secs().unwrap() - clip.duration_secs()).abs() < 0.1 + d * 0.02);
            src.seek(1.0).unwrap();
            let mut buf = vec![0.0f32; 8192];
            let got = src.read(&mut buf).unwrap();
            assert!(got > 0, "{}: nothing after seek", p.display());
            let streamed: Vec<f32> = buf[..got * 2].chunks(2).map(|f| 0.5 * (f[0] + f[1])).collect();
            let energy: f32 = streamed.iter().map(|v| v * v).sum();
            if energy < 1e-3 {
                continue; // a quiet passage says nothing about alignment
            }
            let sr = clip.sample_rate as i64;
            let centre = sr; // 1 s
            let tol = sr / 100; // ±10 ms
            let mut best = (f32::MIN, 0i64);
            for lag in -tol..=tol {
                let start = centre + lag;
                let mut num = 0.0f32;
                let mut da = 0.0f32;
                let mut db = 0.0f32;
                for (i, s) in streamed.iter().enumerate() {
                    let j = start + i as i64;
                    if j < 0 || j as usize >= clip.samples.len() {
                        break;
                    }
                    let c = clip.samples[j as usize];
                    num += s * c;
                    da += s * s;
                    db += c * c;
                }
                let r = if da > 0.0 && db > 0.0 { num / (da * db).sqrt() } else { 0.0 };
                if r > best.0 {
                    best = (r, lag);
                }
            }
            assert!(best.0 > 0.9, "{}: streamed audio after seek does not match the clip (r={}, lag={})", p.display(), best.0, best.1);
            worst_lag = worst_lag.max(best.1.abs());
            seeks += 1;
        }
        eprintln!("real files: {n} probed and decoded, {seeks} seeks checked, worst seek offset {worst_lag} samples");
        assert!(n >= 10);
    }

    #[test]
    fn extension_check() {
        assert!(is_supported(Path::new("a.OGG")));
        assert!(is_supported(Path::new("a.wav")));
        assert!(!is_supported(Path::new("a.txt")));
        assert!(!is_supported(Path::new("noext")));
    }
}

