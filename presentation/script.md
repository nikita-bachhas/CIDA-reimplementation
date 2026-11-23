# CIDA reimplementation, 5-minute talk script

Deck: `presentation/slides.html`. Open it in Chrome or Edge, press **F** for full screen and **P** for the presenter window with these notes and a timer.

How to use this script.

- Rehearse it until you can say each slide in your own words. The prof marks down reading from a script.
- **[click]** means press the right arrow once, which reveals the next part of the slide.
- Times are targets. The total is 4:55, which leaves 5 seconds of slack.
- Text in square brackets is for the team to fill in before the talk.

| Slide | Topic                            | Time | Running |
| ----- | -------------------------------- | ---- | ------- |
| 1     | Title                            | 5 s  | 0:05    |
| 2     | The problem                      | 25 s | 0:30    |
| 3     | Earlier work and its limits      | 35 s | 1:05    |
| 4     | How CIDA works, at home          | 25 s | 1:30    |
| 5     | How CIDA works, in the new place | 25 s | 1:55    |
| 6     | The baseline                     | 30 s | 2:25    |
| 7     | Why the baseline fails           | 40 s | 3:05    |
| 8     | Four changes                     | 35 s | 3:40    |
| 9     | Results                          | 30 s | 4:10    |
| 10    | Limitations                      | 25 s | 4:35    |
| 11    | Future work                      | 20 s | 4:55    |

The reimplementation, slides 6 to 9, takes 130 seconds, the largest share of the talk. The total is 4:55, which leaves 5 seconds of slack.

---

## Slide 1. Title (5 s)

Hello, I am Ryan. Our team rebuilt a recent vision method, called CIDA, from scratch.

## Slide 2. The problem (25 s)

Picture a self-driving car camera trained in the city and then shipped to the countryside. Familiar things like cars now look different, and things like tractors it has never seen. The upgrade has three limits.

**[click]** The city photos are gone,

**[click]** the new photos have no labels,

**[click]** and we get one example photo per new thing.

## Slide 3. Earlier work and its limits (35 s)

Earlier research solves only half of this.

**[click]** Domain adaptation copes with the new look, but it needs the old photos and lumps all new things into one unknown pile. So even its best method gets only 28.6 percent on new things.

**[click]** Class-incremental learning adds new names, but most of it ignores the new look and needs old photos or many labels.

**[click]**

CIDA is the only approach that ticks all three boxes, and it reaches 70.8 percent with one photo per new thing.

## Slide 4. How CIDA works, at home (25 s)

This is how CIDA prepares before leaving home.

**[click]** The extractor embeds each image into a feature vector, and the classifier maps that vector to logits over the source classes.

**[click]** Training squeezes each class into a tight bubble.

**[click]** Fake points between the bubbles teach the answer none of the above.

**[click]** And only a summary of the bubbles travels, never the photos.

## Slide 5. How CIDA works, in the new place (25 s)

**[click]** In the new place, CIDA adapts a copy of the extractor and adds a classifier for new things.

**[click]** Every class gets a magnet. New things take their magnet from one example photo.

**[click]** Each photo takes its nearest magnet's label, and the most confident ones teach the model.

**[click]** Samples from the bubbles stop it forgetting old classes.

## Slide 6. The baseline (30 s)

Here is our baseline: CIDA exactly as the paper describes it, with none of our four changes. There was no public code, so this is our own build.

**[click]** On digits it lands 11 to 26 points below the paper.

**[click]** On office photos, 3 to 39 points below, and on the new classes it is far worse: 3 to 36 percent, where the paper reports 62 to 88.

## Slide 7. Why the baseline fails (40 s)

The method has no safeguard against its own mistakes.

**[click]** Only one new-class image is labelled, so the model labels the rest itself.

**[click]** Training pulls each image towards that guide.

**[click]** So features drift to the wrong guide, and the next labels are worse.

**[click]** On digits, known images drift into the new classes.

**[click]** On office photos, the other way round.

**[click]** This is the method, not our code. The gap survives even the best epoch.
## Slide 8. Four changes (35 s)

We apply four changes, one for each cause.

**[click]** First, recalibrate on the new photos, so the model learns what normal looks like.

**[click]** Second, the bridge converts features into the form the frozen old classifier understands. At the start it should hand them back unchanged. The paper's version changes them by 13 percent. Ours does not.

**[click]** Third, it learns only from the fifth of its own labels it is surest about.

**[click]** Fourth, the image network already knows general vision, so we keep it fixed.
## Slide 9. Results (30 s)

Read it left to right.

**[click]** With our four changes, MNIST to USPS matches the paper,

**[click]** USPS to MNIST matches it too,

**[click]** and SVHN to MNIST comes out higher.

**[click]** On office photos the four changes fall about six points short,

**[click]** on both splits. Hand-picking the one labelled image lifts both past the paper overall, though new classes stay lower.

## Slide 10. Limitations (25 s)

Four limits in this reimplementation.

**[click]** We did not run VisDA at all; the GPU budget did not stretch.

**[click]** New classes are still below the paper, except on SVHN to MNIST.

**[click]** Our changes depart from the paper, so we report both versions.

**[click]** And two parts of the paper we did not implement.
## Slide 11. Future work (20 s)

For the future work, **[click]** we will run VisDA, **[click]** close the new-class gap, **[click]** and find a label-free way to choose the epoch. Lessons that carry over: recalibrate on the new data first, start the bridge as an exact pass-through, and learn only from confident labels. Thank you.

## Where each number comes from

"Report" means the team report, `report/` on the `report` branch (PDF of commit `d153099`). All of our results are the mean of 5 seeds, at epoch 15 on Digits and epoch 20 on Office-31, chosen without target labels. VisDA was not run.

| Slide | Number                                                                                                   | Source                                                                                                                                                                  |
| ----- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3     | 28.6, 62.1, 66.3, 63.0, 70.8                                                                             | Paper, Table 3 (Office-31 average, accuracy on new classes). Earlier methods use 5% labelled new photos.                                                                |
| 3     | Check marks                                                                                              | Paper, Table 1                                                                                                                                                          |
| 6     | 11 to 26 (Digits), 3 to 39 (Office) points below                                                        | Report, Table 3 (baseline, `preset=vanilla`)                                                                  |
| 6     | New classes 44.2 to 59.0 (Digits), 3.2 to 36.4 (Office) against 45.7 to 71.6 and 61.8 to 88.4           | Report, Table 3, PRIV columns                                                                                 |
| 6     | Bars: 72.5 vs 47.0, 69.4 vs 49.6, 46.4 vs 35.1, 72.2 vs 54.3; 15 transfers                              | Report, Table 3 (Office bar is A→D in the 10/20 setting)                                                      |
| 6     | 10 settings unspecified, 10 vs 11 Office classes, 3σ test keeps almost every point                      | Report, Table 2 and Section 3.3                                                                                                                                         |
| 7     | The feedback loop; best epoch between 0 and 12                                                           | Report, Section 5.1, "The model trains on its own mistakes"                                                                                                             |
| 7     | 38 to 44% (37.5 to 43.9), 57 to 60%, up to 94%, 10% (10.4) and single-class collapse                     | Report, Sections 4.2 and 5.1                                                                                                                                            |
| 7     | 1 to 19 points on Office-31 at the best epoch                                         | Report, Section 5.1, "Our reproduction or the method"                                                                                                                   |
| 8     | 74 to 78% → 14 to 21%                                                                                   | Report, Section 5.1, cause 1 (Digits M→U)                                                                                                                              |
| 8     | 13% → 0                                                                                  | Report, Section 5.1, cause 2, and Section 3.2, change 2 (exact identity at the start)                                                                                   |
| 8     | 52.8 → 83.4                                                                                             | Report, Section 5.1, cause 3 (Digits M→U, shared accuracy at epoch 30, other changes on)                                                                               |
| 8     | 27.0 to 33.1 → 72.4, approximately 900 photos                                                           | Report, Section 5.1, cause 4 (Office A→D, nearest-guide accuracy on private classes)                                                                                   |
| 9     | Paper columns                                                                                            | Paper, Tables 2 and 3, as listed in report Table 3                                                                                                                      |
| 9     | As written: 47.0/54.0, 49.6/59.0, 35.1/44.2; Office 10/20 55.6/29.6, 20/31 50.1/17.8 | Report, Table 3, Baseline columns |
| 9     | With our changes: 71.6/57.5, 68.3/54.4, 58.2/52.6; Office 10/20 65.1/42.1, 20/31 65.8/30.1. Hand-picked: 83.7/75.5, 77.1/71.0, 60.4/59.1; Office 10/20 72.2/61.8, 20/31 77.3/62.3 | Report, Table 3, Improved and Handpicked columns. Hand-picked fixes the labelled image across seeds; on Office it also adds label propagation and longer source training, and the 10/20 figure is one seed. Report prose gives the averages as 72.2/61.8 (10/20) and 77.3/62.3 (20/31) |
| 9     | Office paper columns 70.9/70.9 (10/20) and 72.4/71.8 (20/31) | Means of the six rows of each setting in report Table 3 |
| 10    | One GPU, six hours, one job at a time                                                                    | `README.md` line 34, CCDS GPU TC2 QoS `normal`; report Section 4.1                                                                                                  |
| 10    | New-class gaps; VisDA not run; Ours-d and A→D→W not implemented                                          | Report, Table 3, Section 3 ("we run the method on Digits and Office-31 only") and Section 5.3                 |
| 11    | Epoch 15 for every task; baseline peaks by epoch 12; the four lessons                                    | Report, Sections 4.1, 5.1 and 6                                                                                                                                         |
