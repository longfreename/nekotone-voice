# Voice Studio

This guide is for changing your voice live and sending it to other programs as a microphone: games, Discord, Teams, OBS. Your real voice never goes through: the other side only ever hears the processed voice, or silence.

Its second mode, [Speak for me](#speak-for-me), replaces your voice with a synthetic one that reads out what you say or type.

Voice Studio is optional. Turn it on in **Settings → Voice changer**; a **Voice** item then appears in the sidebar.

**Time:** 10 minutes, including the one-time virtual cable setup.

## Step 1. Give other programs a way to hear it (Windows)

Windows cannot create a new microphone without a signed driver, so Voice Studio sends its voice into a free *virtual audio cable*, and your other program listens to the cable.

1. Install **VB-CABLE** (free) from [vb-audio.com/Cable](https://vb-audio.com/Cable/) and restart the PC if it asks.
2. Open **Voice → Setup**. Under **Send to other apps as a microphone** Voicekit shows the cable it found and the exact name to choose in the other program, usually **CABLE Output (VB-Audio Virtual Cable)**. The copy button copies that name.
3. In Discord, Teams, OBS or your game, choose that name as the microphone.

VoiceMeeter and Virtual Audio Cable work too; Voicekit detects them by name. It never installs drivers itself.

## Step 2. Pick your microphone and a voice

1. In **Setup**, choose your microphone. The noise gate is automatic; set a threshold if your room is noisy.
2. Open **Voices**. Every voice has a **Preview** button: it plays a sample through the voice. **Record a sample (15 s)** lets you preview with your own voice instead, and measures your voice for the voices that adapt to you.
3. Click a voice to **go live**. The bar at the top shows **ON AIR**, the input and output levels, whether the gate is open, your pitch and the voice's pitch, and the delay.

| Group | Voices |
|---|---|
| You | Confident me, Relaxed me, Storyteller me, Professional me, Younger me, Older me, Anonymous me |
| Speakers | Emma, Olivia, Chloe, Margaret, Nina (women); Liam, James, Victor, Walter, Tyler, Neil (men); with an accent: Sophie, Henry (British), Mia, Jack (Australian), Siobhan (Irish), Callum (Scottish), Daisy (Southern US), Arjun (Indian), Rhys (Welsh) |
| Transform | Female, Male, Child, Mouse, Giant, Deep Narrator |
| Creatures | Dragon, Demon, Alien, Ghost, Robot, WuuWuu, Underwater |
| Spaces & FX | Cathedral, Choir, Radio, Telephone, Megaphone, Studio |

Voices marked **keeps your voice** (the **You** group, Cathedral, Choir, Radio, Telephone, Megaphone, Studio) leave your voice recognisable. All the others disguise it.

### You

The **You** group is *you*, with a twist: the same person, more confident, calmer, telling a story, on an important call, younger or older. They are built from the six **voice controls** (see Step 3) on your own voice, and level your voice to a normal speaking level without changing its colour. **Anonymous me** is harder to recognise but still natural (a different pitch and head size, your personal melody evened out); it is not a full disguise, for that use a Speaker.

### Speakers

**Speakers** are other people built on your voice: women and men of different ages and colours. Each one has its own speaking pitch and vocal-tract size (so it lands on the same voice whoever talks), its own melody (steadier or livelier), and what makes a voice recognisable beyond pitch: Margaret and Nina are breathy, Walter is gravelly with a slight tremor, Neil is nasal, Victor is a deep steady bass, Chloe and Tyler are teenagers. Your words and timing stay yours: only the voice changes. To colour the accent too, pick one under **Accent** (below).

### Accent

**Tune → Accent** colours your own voice towards an accent as you speak, on any voice (a Speaker, Female, your own voice with no other change…):

| Accent | Vowels | r | Melody |
|---|---|---|---|
| British RP | LOT further back, GOAT further forward | (to be taken out; not yet working) | phrases fall, a narrower range |
| Australian | TRAP raised, GOOSE and FOOT fronted | (to be taken out; not yet working) | phrases rise (uptalk), livelier |
| Irish | STRUT and TRAP further back | kept | a lilt, rising |
| Scottish | GOOSE strongly fronted, low vowels central | kept | gently rising |
| Southern US | TRAP raised, GOOSE fronted | kept | drawled, wide range, falling |
| Indian | TRAP raised, STRUT centralised | kept | a syllable-rate lilt |
| Welsh | GOOSE a little fronted | (to be taken out; not yet working) | a strong sing-song lilt |

How it works: every 5 ms the engine finds the first three formants of your voice (the resonances that make one vowel sound different from another), recognises which kind of vowel it is from where they sit (scaled to your vocal-tract size, so it works for any voice), and moves the ones that the accent pronounces differently. An "r-coloured" vowel (the r in *bird*, *car*) has its third formant pulled down near the second; accents that drop the r are meant to push it back up, but this part does not work yet (the two formants are too close for the engine to tell apart), so the r stays. The melody of each phrase (from the first voiced sound after a pause) rises, falls or lilts the way the accent does. **Accent amount** in the Voice block (signal chain) sets how strong it is.

The vowel moves are partial: through the whole voice stage about half of each move shows (a formant's energy is partly in the excitation, which cannot be moved). It is an accent *colour*: it changes how your vowels and melody sound, not which vowel a word uses (Australian "day" sounding like "die" needs the words). For a full accent, use **Speak for me**, which re-speaks your words.

### Voices that adapt to you

Voices marked **adapts to you** (all the Speakers, and Female, Male, Child, Mouse, Giant, Deep Narrator, Dragon, Demon, Alien, Ghost, WuuWuu) are defined by where they end up, not by a fixed shift: Female speaks at about 205 Hz through a smaller vocal tract, Dragon at about 72 Hz through a large one, whoever is talking. Voicekit measures your speaking pitch, how much you move it, and the size of your vocal tract as you talk, and works out the shift that takes *your* voice there. So a man and a woman choosing Female come out at the same pitch (measured: within a semitone), where a fixed +5 semitones used to leave them an octave apart. Your own melody is kept; only the overall level of your voice moves, slowly, so it never wobbles inside a sentence.

- It learns within a few seconds of speech. **Record a sample** also tunes the voices to you (**Tuned to you · 112 Hz** next to the button), so they are right from your first word; the × forgets it. They keep listening either way, so if someone else takes the microphone the voice follows them.
- **Your voice** under **Engine** in **Tune** shows what has been measured (pitch, range, tract size; *learning…* for the first seconds).
- **Pitch** and **Formant** still work on these voices: they move the voice up or down from its target.
- A **Timbre match** block keeps the level, body and brightness on the voice's target too, so it sounds the same whether you mumble or project, on any microphone. You can add it to your own voices from the signal chain.

### Your voice identity

Voicekit keeps a small model of your voice, on this PC only (`voice-profile.json` in the data folder):

| Layer | What | Measured |
|---|---|---|
| Physiology | speaking pitch, vocal-tract size | live, as you speak, and from your sample |
| Behaviour | how much your pitch moves, speaking rate | pitch live; rate from your sample |
| Style | speech level, body, brightness, breathiness | from your sample |

**Record a sample** (or **Tune voices to me**) measures all of it; **Your voice** in **Tune** and the **Tuned to you** badge show it. When your microphone is quiet (below −40 dB while you speak) it says so: quiet speech is measured less well, so raise the microphone's level in Windows or speak closer to it.

- **Keep learning my voice** (next to the sample, off until you turn it on): when you stop the voice changer, what it measured refines your saved identity. Only when it was clearly you (the pitch within 5 semitones and the tract within 12 % of what is saved), so someone else at the microphone or you doing a character voice never changes it. Newer sessions count more over time.
- Shouts, laughs, squeals and vocal fry are *events*, not how you normally speak: they never move your identity (measured: a talker with a shout or a laugh every 4 seconds keeps his pitch to within a tenth of a hertz). What lasts longer than about a second and a half is taken as the new normal, so a new speaker is still followed.
- The × on **Tuned to you** forgets all of it.

## Step 3. Shape it

**Tune** has five knobs: **Pitch**, **Formant** (the size of the "head" the voice seems to come from: turn it up for smaller and brighter, down for bigger and darker), **Character** (how strong the voice's effects are), **Space** (reverb or echo) and **Output**. Drag up or to the right to turn a knob up (down or left to turn it down; hold Shift for fine steps), use the wheel, or double-click to reset.

**Voice controls** move *who you sound like*, not the sound. Each slider goes from −100 to +100 (double-click resets it; **Reset** resets all six):

| Control | − | + | What it moves |
|---|---|---|---|
| **Confidence** | less | more | a little lower and steadier, a wider melody, more presence and level |
| **Warmth** | cooler | warmer | more body below 300 Hz, less air above 5 kHz |
| **Clarity** | softer | clearer | more presence around 3 kHz and air, less boxiness |
| **Age** | younger | older | younger: higher, a smaller vocal tract; older: lower, breathier, rougher, a slight tremor |
| **Gender** | masculine | feminine | up to 4 semitones and 8 % of vocal-tract size either way, with the melody and tone that go with it |
| **Energy** | calmer | livelier | how much your pitch moves, and a little level |

They work on any voice with a Voice block: on a voice that adapts to you they move its target, on your own voice they move you. Measured on a test voice: Gender +100 raises it 3.97 semitones and shrinks the vocal tract; Confidence widens the melody from 1.9 to 2.3 semitones and Energy −100 narrows it to 1.35; Warmth tilts the tone about 3.7 dB towards the body; Clarity adds 3.2 dB of presence; Older lowers the harmonics-to-noise ratio from 21 to 12.5 dB. **Save as** keeps the result as your own voice.

A **realism guard** works on every voice:

- **Pitch glitches:** when the pitch tracker misreads a noisy or creaky moment (often by an octave), that moment is sung at your recent pitch instead of jumping; a real jump that lasts passes. On a noisy real recording it took about a third of the pitch glitches out of the male voices.
- **Metallic ringing:** a narrow peak between 1.5 and 10 kHz that stays on one frequency for a quarter of a second (voices move, ringing does not) is cut by a narrow notch, up to 8 dB; the harmonics of the voice's own pitch are never touched. In a test it took a ring down by 8 dB and left the voice around it within half a decibel; on the built-in voices it rarely has anything to do (at most half a decibel, on Giant and Dragon).

Below them, the **signal chain** lists every block of the voice (gate, pitch and formant, growl, reverb, …) with all its settings, live. Add, move or remove blocks, then **Save as** to keep your own voice under **My voices**.

## Sound pads

**Pads** (the tab next to Tune) is a soundboard for streams and calls: short sounds played into the same output as your voice, so the other program hears them too.

1. **Add sounds…** picks audio files (up to 12; each plays up to 20 seconds). Each pad gets a name you can change, a level, and a shortcut: **Ctrl+Alt+1** to **Ctrl+Alt+9**, which work while a game or Discord has the focus (like **Ctrl+Alt+M** for mute; the **global shortcut** setting in Setup turns them all off).
2. Go live, then click a pad or press its keys. Pressing a pad that is playing starts it again; up to four play at once. **Stop all** stops them.
3. **Duck my voice** (0 to 30 dB, 10 by default) lowers your voice while a pad plays, so the sound is heard over you; it comes back over about a third of a second.

Pads go through the limiter and the mute like your voice: they can never clip the output, and **Mute** silences them too. They are files, not your microphone, so nothing about the voice changer's privacy changes. The first press of a pad reads the file; after that it starts at once.

## Mute and safety

- **Mute** (or **Ctrl+Alt+M**, which also works while a game or Discord has focus) sends silence at once. Press it again to come back.
- There is no path for your unprocessed voice to reach the output, and when the voice stage is not ready the output is silent, not dry.
- A limiter keeps the output below the ceiling in **Setup**, so a shout cannot clip.
- **Hear yourself** plays the voice to your headphones. Use headphones: on speakers the microphone hears itself and howls.

## Delay

The processing itself adds about 26 ms. (It was 12 ms before 0.4.2; the voice stage now keeps a whole pitch period of a low voice in each grain, which is what stopped voices under about 200 Hz sounding buzzy.) The rest of the delay you see in the bar comes from your audio devices' buffers (on some USB interfaces 60 ms or more). For the lowest delay use the device's own driver settings for a smaller buffer, and a microphone and output on the same device.

## Speak for me

Speak for me is the other mode of Voice Studio (the switch at the top). Instead of changing your voice, it **replaces** it: you talk or type, Voicekit writes down what you said, and a natural synthetic voice says it to the other program. The other side hears the synthetic voice or silence, never you. Use it when your voice must not be recognised at all, when you cannot speak aloud, or to talk in English when you speak another language.

How it works, on this PC:

1. A speech detector listens to the microphone and waits for the end of your sentence (a short pause).
2. Whisper writes the sentence down (or translates it into English).
3. Kokoro, a small open text-to-speech model, says it in the voice you picked, sentence by sentence; an effect from Voice Studio can be added on top.
4. The voice goes to the virtual cable (or an output you choose), exactly like the voice changer's.

The microphone is used only by steps 1 and 2. Nothing in the program copies it to the output.

### Set it up

1. Open **Voice**, choose **Speak for me** at the top.
2. The first time, press **Download the voices** (353 MB, once; it is not part of the installer). Listening also needs a Whisper model: the one chosen in **Settings** (`whisper-base` to start). Without one you can still type.
3. **Setup** is the same panel as the voice changer's: the microphone, **Virtual microphone** (VB-CABLE, see Step 1 above) or an output device, and **Hear yourself**.
4. In **Voices**, press the play button on a voice to hear it, click a voice to choose it, and press **Go on air**.

### Talk or type

- Speak normally and pause at the end of a sentence. After the pause the voice starts, and the **Live** log shows what you said (in italics) and what the voice said, with the measured delay.
- Type in the box at the bottom of **Live** and press **Enter** (Shift+Enter for a new line). Typing works while the microphone is off.
- **Skip** stops the line that is playing (the next one starts); **Clear** stops, drops everything queued and clears the conversation. They are greyed out when there is nothing to skip or clear. **Listening to you** turns the microphone off and on.
- **Mute** (or **Ctrl+Alt+M**) sends silence at once, as in the voice changer.

### Voices, speed and effects

There are 28 English voices, American and British, female and male. The gallery shows the eight best rated (Heart, Bella, Nicole, Michael, Fenrir, Puck, Emma, George); **All voices** shows the rest. The letter next to a name is the quality grade given by Kokoro's authors (A is best).

| Control | What it does |
|---|---|
| **Speed** | 0.7× to 1.4×; double-click resets to 1× |
| **Effect** | Any Voice Studio voice on top of the synthetic one: Cathedral, Radio, Telephone, Dragon… **Clean** is the plain voice |
| **Accent** | Any voice can speak in any of these: American, British (RP), Australian, London, Irish, Scottish, Southern US, Indian English, or **the voice's own** |
| **Sound like me** | Moves the voice to your own speaking pitch and vocal-tract size (measured when you **Record a sample** in the voice changer), so a voice with an accent sounds more like you |
| **Translate to English** | You speak any language Whisper knows; the voice says it in English. **I speak** tells Whisper which language, which helps with short sentences |

The voices speak English. With **Translate** off, speak English; with it on, speak your own language.

Accents are made by rewriting the voice's pronunciation (its phonemes) with the features that make each accent recognisable: Australian and London diphthongs, Scottish and Indian tapped r, Irish and Indian "t" for "th", the Southern "ah" for "I", retroflex t and d in Indian English, and so on. Kokoro was trained on American and British speakers, so an accent is a strong impression rather than a native speaker; pick the voice whose sound you like best and try a few accents with the play button.

**End of a sentence** is how long you pause before it speaks. With **Auto** ticked (the default) it learns this from you as you talk: it watches your pauses and waits through the short ones you make inside a sentence, so it answers sooner when you talk quickly and waits longer when you stop to think. The slider is where it starts, and shows the value in use ("auto · 410 ms"). Untick Auto for a fixed wait.

### Speak in your own voice

**My voices**, the first card in **Voices**, clones your voice from about fifteen seconds of you talking. Speak for me then says your words (typed, or translated from another language) in that voice. You can keep several, each with its own name: one per microphone, a calm one and an energetic one, or the same voice from a better recording. The model is **Chatterbox-Turbo** by Resemble AI, and it runs on this PC.

Clone only **your own voice, or a voice whose owner agreed**. The card asks you to confirm that once.

1. In **Voices**, on the **My voices** card, press **Download the voice-clone model** (about 1.3 GB, once).
2. Tick **My own voice, or the owner agreed**.
3. Type a name for the voice (empty: "My voice"), then press **Record 15 s** and read something aloud in your normal voice, in a quiet room. When the recording ends, the voice is made (the first time, loading the model adds a few seconds). **Use a file…** makes the voice from a recording instead: 5 to 15 seconds of clear speech by one person. **From last sample** makes a voice from the recording you made last, without recording again.
4. Each voice gets its own card. Press its play button to hear it, then click the card to choose it and press **Go on air**.

Hover over a voice's card for **Rename** (the pencil) and **Delete** (the bin). Renaming keeps it chosen; deleting removes only that voice, and your recorded sample stays until you delete it in the voice changer. Under each name: how many seconds of speech it was made from, and when (hover for the recording it came from).

Each voice is turned into a small voice file once, in `voice-clone` in Voicekit's data folder (the first one is `mine.nkvoice`), and speaking reuses that file. On a PC without a suitable graphics card, the card says **slow on this PC**: a cloned voice then takes a few seconds per sentence, fine for typing and slow for live talk.

| | Your voice | Kokoro voices |
|---|---|---|
| **Speed**, **Accent** | Not available (greyed out): your voice keeps the pace and accent of your recording | Available |
| **Effect**, **Sound like me** | Effect works; Sound like me is not needed | Both work |
| First words after you stop talking | About 2–3 s for a short sentence: the model writes the whole sentence as sound tokens before it speaks (measured: 2–3 s of speech made in 1.6–2.9 s) | About 1.2 s |
| Needs | Any PC; the sentence model runs on the CPU (a graphics card speeds up the last step) | The CPU is enough |

Switching between one of **My voices** and a Kokoro voice while on air restarts Speak for me for a moment, because your voice uses a different model.

What to record: a steady voice, no music or other people, the microphone at its usual distance. The model copies what it hears, including room echo and a noisy microphone, so a cleaner sample gives a cleaner voice.

### Change my voice (neural)

**Change my voice (neural)**, a switch next to Accent in **Voices**, keeps *your own words, timing and delivery* and changes only the voice. Nothing to type: each thing you say is spoken again, in the chosen voice, about a second after you stop, and the transcript shows what you said (Whisper writes it down after the phrase is on its way, so the voice never waits for it). **Keep my pauses** (on by default) leaves the pause you left before each phrase, up to 2 s. It works with every voice in the gallery and with **My voices**. It needs the voice-clone model (see above).

- The first time you use a Kokoro voice with it, the voice is learnt once (a few seconds: Kokoro reads a paragraph, which is turned into a voice file kept for next time).
- It sounds like a real person, not an effect: measured on a recorded voice turned into three Kokoro voices, the result was 0.87–0.93 similar to the target voice (and 0.51–0.68 to the original speaker) on a speaker-recognition scale, and Whisper heard the same words in 33 or 34 of 34. Ten seconds of speech are converted in about 0.8 s.
- Unlike the voice changer (Step 2), it is not instant: it waits for the end of each phrase. Use the voice changer for live, word-for-word sound; use this when the voice has to be convincing.
- **Effect** and the output settings still apply, so a Cathedral or Radio effect works on the new voice too. **Speed** and **Accent** are greyed out: the voice keeps your own pace and pronunciation.
- **Drop "um" and "uh"** and **Mask swearing** work here too: Whisper listens to each phrase for those words (a few tenths of a second more), fillers are cut out of your phrase and swear words bleeped in what the voice says. The live log then shows your words instead of "(your words)".
- As with every voice: only use a real person's voice with their agreement.

### Speech options (Live)

| Option | Default | What it does |
|---|---|---|
| **End of a sentence** | 275 ms | How long a pause ends a sentence. Shorter answers sooner; longer lets you pause to think mid-sentence. The default comes from measuring real speech: most pauses inside a phrase are under 250 ms, clause breaks 250–400 ms, sentence ends over 500 ms |
| **Drop "um" and "uh"** | on | Filler words and bracketed noises such as [laughs] are not spoken. With **Change my voice**, they are cut out of what you said before the voice says it |
| **Mask swearing** | off | Common swear words are spoken as "bleep". With **Change my voice**, they are bleeped in what the voice says |
| **Pause the mic while it speaks** | off | Turn on if you use speakers instead of headphones, so it does not hear and repeat itself |

### How long it takes

From the end of your sentence to the first word of the voice, measured on the development PC (a sentence of about two seconds, `whisper-base`):

| Whisper runs on | Fastest | Typical | Slowest |
|---|---|---|---|
| the CPU | 1.20 s | 1.27 s | 1.47 s |
| the GPU (DirectML) | 1.00 s | 1.18 s | 1.52 s |

The time is the pause that ends the sentence, Whisper (0.25 to 0.5 s), and the voice's first words (0.3 to 0.55 s). A long reply does not take longer to start: the rest of it is made while the first part plays. The voice itself always runs on the CPU, at about four times real time. A larger Whisper model is more accurate but slower.

### Licences

- Voices: **Kokoro-82M** by hexgrad, Apache-2.0 (ONNX export by onnx-community).
- Pronunciation: the **misaki** dictionaries by hexgrad, Apache-2.0. Words that are not in them are built from known parts or read with Voicekit's own letter-to-sound rules; no GPL software (such as espeak) is used.
- Listening: OpenAI **Whisper**, MIT.
- Your own voice: **Chatterbox-Turbo** by Resemble AI, MIT (the ONNX export `ResembleAI/chatterbox-turbo-ONNX`). Resemble's PerTh watermark, which their Python package adds to its output, is not added here.

### Problems

| Symptom | Cause | Fix |
|---|---|---|
| It speaks too early, in the middle of your sentence | Your pauses are longer than **End of a sentence** | With **Auto** on, keep talking for a minute so it learns your pauses; or untick Auto and raise the slider to 600 ms or more |
| It repeats itself | Speakers, not headphones: the microphone hears the voice | Use headphones, or turn on **Pause the mic while it speaks** |
| A name comes out wrong | The name is not in the dictionaries | Type it the way it sounds ("Shiv-awn" for Siobhan) |
| Nothing happens when you talk | The microphone is off, or no Whisper model | **Listening to you** in Live; download the model it names |
| **Make my voice** says the sample is too short | Less than 5 seconds of sound after silence is trimmed | Record again and keep talking for the whole 15 seconds |
| Your voice sounds muffled or echoey | The sample was recorded in a big room or far from the microphone | Record again closer to the microphone, in a quieter room |
| Your voice takes a few seconds to answer | Your voice writes each sentence before it speaks (about real time) | Short sentences answer sooner; use a Kokoro voice for the quickest replies |
| **Change my voice** is greyed out | The voice-clone model is not downloaded | Download it on the **My voices** card |
| **Make my voice** failed with an error (0.4.0) | The voice encoder failed on the graphics card | Fixed in 0.4.1: it now runs on the CPU |

## Common problems

| Symptom | Cause | Fix |
|---|---|---|
| The other program hears nothing | It is listening to the wrong device | Choose **CABLE Output (VB-Audio Virtual Cable)** as its microphone |
| No cable is found | VB-CABLE is not installed, or the PC was not restarted | Install it, restart, press **Check again** |
| The voice cuts in and out | The gate is closing on quiet speech | Lower the gate threshold in **Setup** |
| Howling | **Hear yourself** is on without headphones | Turn it off or use headphones |
| Robotic or warbly sound | A very large pitch or formant change | Move **Pitch** and **Formant** closer to the middle; big changes sound more processed |
