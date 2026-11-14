# 5-minute pitch: 2 slides, 2–3 speakers

Audience: non-technical stakeholders. Goal: they understand the problem, what our system does, and why we trust it.
Rule of thumb: no model names, no formulas, one number per point.

---

## Who says what

| | 3 speakers | 2 speakers | Time |
|---|---|---|---|
| Slide 1, part 1: the problem | A | A | 1:15 |
| Slide 1, part 2: how we decide | B | A | 1:15 |
| Slide 2: how we tested it, and the result | C | B | 2:00 |
| Closing line + questions | C | B | 0:30 |

---

## Slide 1: Saying yes to the right bookings

**On the slide:**
- Two hotels, 227 rooms, and about 1 in 3 bookings is cancelled
- Accept too few → empty rooms. Accept too many → guests turned away at the door
- Our system: for every booking request, *yes* or *no*, in real time

**A, the problem (1:15):**
- "Our two hotels have 124 and 103 rooms. About 36% of all bookings are cancelled, so a hotel that only sells as many rooms as it has ends up half empty."
- "But sell too many, and guests arrive to a full hotel. In this challenge, turning a guest away costs 2.5 times what they would have paid."
- "At the busiest times, almost twice as many guests want a room as there are rooms. So the real question is *which* bookings to accept, and how many."

**B, how we decide (1:15):**
- "We learned from 78,000 past bookings. For every new request, the system estimates how likely this guest really is to show up. It knows that some booking types, deposits and channels cancel far more often than others."
- "Then it keeps a running count for every night: how many guests we expect to actually turn up."
- "A booking is accepted if, on every night of the stay, the expected guests still fit, with 10% overbooking, a margin we tested on last year's data. Bookings that pay above the usual price get a little more room."

---

## Slide 2: Tested on last year, before it counts

**On the slide:**
- A dress rehearsal on 2016: four different periods, as if live
- 200,000+ combinations tested on a university computing cluster in 14 minutes
- Result: we capture **74%** of the achievable extra revenue, up from **61%** in our first version
- 2017 official result: **[fill in after the evaluation]**

**Suggested picture:** one horizontal bar from 0% to 100%.
- 0% = accept no new bookings (play it safe)
- 61% = our first version
- 74% = our final system
- 100% = knowing in advance who will show up (impossible)

**C, how we tested it, and the result (2:00):**
- "We replayed 2016 booking by booking, as if it were happening live. The system only knew what it would have known on that day."
- "We tested four different periods. And we always judged a setting on a period it hadn't been tuned on, so it can't just memorise the answers."
- "On a university computing cluster we tried more than 200,000 combinations of prediction methods and decision rules. That took 14 minutes."
- "Compared with knowing in advance who shows up, we capture about 74% of the extra revenue that's possible, up from 61%. For January to August 2016, that's about €260,000 more net revenue, roughly 7%."
- "The biggest win wasn't a smarter prediction. It was the decision rule. Our predictions were a bit too optimistic about guests showing up, so the hotels looked full too early. Allowing 10% overbooking fixed that."

**Closing (0:30):**
- "So: a simple, tested rule that fills rooms without turning guests away. Our official 2017 result is [fill in]."

---

## Translating our work into plain language

| What we say in the team | What to say to stakeholders |
|---|---|
| Cancellation model, P(show) | "A score for how likely the guest really shows up" |
| Logistic regression + LightGBM blend | "Two standard prediction methods, averaged" |
| Features | "Information about the booking: how far ahead it was made, deposit type, price, channel…" |
| Training data | "78,000 past bookings from 2015–2016" |
| Calibration / the model is biased | "Checking that '30% chance' really means 30 out of 100. Ours was a bit too optimistic" |
| Overbooking factor `alpha` = 1.1 | "We accept up to 10% more expected guests than we have rooms" |
| Price tilt `beta` | "Better-paying bookings get a little more room" |
| Walk penalty 2.5× | "Turning a guest away costs 2.5 times their stay" |
| Replay / backtest | "A dress rehearsal on last year's bookings" |
| Rolling windows, held-out check | "Tested on periods it wasn't tuned on" |
| Hyperparameters, grid search | "Fine-tuning knobs, and trying every combination" |
| HPC / cluster | "A university computing cluster" |
| Share of the possible gain | "How much of the achievable extra revenue we capture" |
| Hindsight ceiling | "As if we knew in advance who shows up" |
| Overfitting | "Learning last year by heart instead of learning the pattern" |
| Data leakage (`booking_id`) | "Information we wouldn't have at decision time, so we didn't use it" |

Avoid on stage: PCA, AUC, log-loss, isotonic, LightGBM, alpha/beta/z, "the model is 87% accurate" (accuracy isn't what we optimise).

---

## Likely questions

- **"Why not just accept every booking?"** In January–August 2016, accepting everything would have ended at minus €1.3M, because so many guests would have been turned away. Accepting nothing new: +€1.2M. Our system: about €3.9M.
- **"Isn't overbooking bad for guests?"** The system only overbooks where cancellations make it safe on average. Every guest turned away counts as a cost of 2.5× their stay, so the system has a strong reason to avoid it.
- **"Why not use more advanced AI?"** We tried several. Beyond the basics, they barely changed the result. The decision rule mattered far more.
- **"How sure are you about 2017?"** It's tested on four 2016 periods, each one judged without being tuned on. 2017 can still behave differently. The official evaluation will tell, and we haven't looked at 2017 data.
- **"What would you do next?"** Put a value on keeping a room free for a better booking later, and retrain the system regularly as new bookings come in.

---

## Numbers to have ready (don't overclaim)

| Number | Value | Careful |
|---|---|---|
| Rooms | 124 + 103 = 227 | |
| Past bookings | 78,132 (Jul 2015 – Dec 2016) | |
| Cancelled | 36% | |
| Combinations tested | 831 prediction setups × 252 decision rules ≈ 209,000, each on 4 periods | |
| Cluster time | 14 minutes | |
| Captured share of possible gain | 74% (first version 61%) | Measured on 2016, not 2017. Slightly optimistic, because the test periods overlap |
| Extra net revenue | about €70,000–260,000 per tested period (+7–13%) | Compared with our first version, not with the hotels' actual past revenue |
| 2017 result | [after the evaluation] | Run it once and don't tune on it |
