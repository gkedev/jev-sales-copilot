# Demo script — "A sales copilot that never generates a word" (3–5 min)

Setup before recording:

```bash
cd sales-copilot && ./run.sh          # http://localhost:8000 — open in Chrome (needed for the mic segment)
```

Have the browser at ~125 % zoom, dark IDE next to it with `copilot/constants.py` open at `SIGNAL_WEIGHTS`.
Verify the header pill says `model jev-1.13.0` and `connected`.

Utterance numbers below match the `#` column of `./run.sh cli --transcript <call>` and the tooltip on the chart.
Speeds: use **4×** for talking over the replay, **20×** to skip ahead. Pause with the ⏸ button; the chart, cards and
signals freeze on the last utterance.

---

## 0. Hook (20 s) — the empty dashboard

> "This is a live sales coach. It listens to a call and, after every single sentence, tells the rep how likely the
> deal is to close, what's going well, what's going wrong, and what to say next. The twist: the model behind it,
> Jev, never generates text. It only answers typed questions — yes/no probabilities, a pick from a list, a score on a
> rubric. Everything you see on screen is arithmetic in code over those answers."

Point at: the weights table at the bottom-left ("that's the whole model of a deal — 18 coefficients").

## 1. The good call (60 s) — `good` at 4×, then 20×

Select **Summit HVAC — discovery-led call that closes**, speed 4×, ▶ Play.

- **#2–#8** — point at *Call stage* flipping `opening → discovery`, and *Next best move* saying **Ask a discovery
  question** → **Quantify the pain**. Talk share stays around 50 %.
- **Pause at #11** ("We tracked it in April. Nine jobs…"): *Live signals* — **Pain identified 🔒** is persisted.
  > "Once Jev is ≥ 70 % sure the pain has been named, code remembers it for the rest of the call — the model doesn't have to."
- Resume, 20× to **#21** ("Okay, this is interesting. What does something like this cost?"): probability jumps
  into the 60s, *Next best move* becomes **Ask about budget**. The rep then does exactly that (#22).
- **Pause at #25** ("Eleven hundred… Greg would need to sign off"): a small dip and an **authority** objection card
  opens. *What moved the number* shows `Objection open −14 pts`.
- Resume to the end (#39 "let's do it"): 🔒 on budget, decision maker, timeline, next step; probability ≈ 90 %;
  coaching shows **Ask for the close** with the highlighted line.

## 2. The bad call (60 s) — `bad` at 4×

Select **Bluewater Plumbing**, ▶ Play.

- **#2** — the rep's 80-word opener: *Transcript* shows `⚠ rep monologue`, rep talk share climbs past 75 %, *Next
  best move* says **Stop talking, ask an open question** — "I've been talking a lot. What's your reaction so far?"
- **Pause at #9** ("Eight fifty a month? … a lot of money for a calendar"): **price objection** card opens, probability
  drops from 30 % to ~20 %, sensitivity panel: `Objection open`, `Buyer engagement`.
  > "Jev didn't decide 'this call is going badly'. It answered 'is the prospect objecting? 0.9', 'what kind? price,
  > 1.0'. The −14 points is a coefficient in the weights table."
- Resume through **#13** (competitor DispatchPro → *Competitor mentioned 🔒*, move: **Address the competitor
  comparison**) and **#15** ("What do you mean, closes the loop?" → *Buyer confused*, move: **Clarify in plain words**).
- Let it run to the end: **#31** "just send me some info" → *Prospect disengaging* lights, engagement `low`, final ≈ 6 %.
  Note the rep never gets *Next step agreed* 🔒 even though the stage is `next_steps`.

## 3. Change a coefficient, not a prompt (40 s) — stay on the finished bad call

- Scroll to *Composite weights*. Change **Objection open** from `-0.14` to `-0.30`, click **Apply weights**.
- The whole chart re-draws lower, instantly; the green toast says **0 Jev calls**.
  > "Every raw answer is stored. Policy — how much an open objection should hurt — is a number, so tuning it is a
  > recompute, not a re-prompt and not a re-inference."
- Click **Reset to defaults**.

## 4. The recovery (40 s) — `mixed` at 20×, pause twice

Select **Ridgeline Electric — rocky start, discovery rescue**, speed 20×, ▶ Play; pause around **#7**.

- Chart dips to ~16 % ("Our tool is fine"), price objection open, competitor 🔒.
- Resume; pause around **#13–#14**: the rep stopped pitching and asked "what does a bad day look like?" — stage back
  to `discovery`, moves are **Quantify the pain**, and the line climbs steadily. Objection card clears on **#21**
  ("That's a fairer way to look at it").
- Resume to the end: ≈ 80 %, pilot agreed on **#33–#35**. Point at the header: **~40 requests, ≈ $0.013 for the
  whole call, ~330 ms per utterance including the network**.

## 5. Live mic (30–45 s) — optional but convincing

Click the **Live mic** tab → **🎙 Start mic** (allow the microphone). Leave the speaker on **Rep speaking** and say:

> "Before I show you anything, can you walk me through how scheduling works for your team today?"

→ *Rep asked discovery question* lights, stage `discovery`, move **Quantify the pain** or **Ask a discovery question**.

Press **S** (or click **Prospect speaking**) and say, as the prospect:

> "Honestly that's way more than we budgeted for. I'd have to think about it."

→ price objection card, probability drops, coaching **Handle the price objection with ROI** with the "say:" line.

Mention on camera: there is no diarization in the browser API, so the speaker toggle is manual; a real deployment
would take a diarized stream from the call platform and everything else stays the same.

## 6. Close (15 s)

> "One request per sentence, thirty-seven questions, one typed answer each, about a third of a second, a third of a
> cent per call. No prompt engineering — the whole behaviour is a constants file you can read, diff and unit-test."

Show `copilot/constants.py` for two seconds and stop recording.

---

### If something goes wrong

- Header says `disconnected · retrying`: the server died — re-run `./run.sh`. The page reconnects by itself.
- Red toast "Jev error": network / rate limit; the replay keeps going, the affected utterance just shows no new signals.
- Chart flat at 25 %: the key is missing — `curl localhost:8000/api/health` should show `"has_api_key": true`.
- Mic button says "Web Speech API not available": you are not in Chrome; use the **Type** tab with the same two lines.
