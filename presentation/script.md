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

**[click]** On office photos, 3 to 25 points below, and on the new classes it is far worse: 9 to 40 percent, where the paper reports 62 to 88.

**[click]** And on VisDA, about 50 points below.

## Slide 7. Why the baseline fails (40 s)

The method as written has no safeguard against its own early mistakes.

**[click]** One image per new class is labelled, so the model labels the rest itself: each image takes the label of its nearest guide.

**[click]** Training then pulls every image towards whichever guide it picked.

**[click]** So the features move towards the guide that was wrong, and the next round of labels is worse. That is why the best epoch is always an early one.

**[click]** On digits, known images drift into the new classes. Only 38 to 44 percent of the first labels are right.

**[click]** On office photos it runs the other way, with up to 94 percent of new-class images called known.

**[click]** On VisDA everything collapses onto a single class.

**[click]** This is the method, not our code. Even at the best epoch, chosen with target labels we would not have, the baseline stays 1 to 19 points short on office photos and 24 to 33 on VisDA.

## Slide 8. Four changes (35 s)

So we made four changes, one for each cause.

**[click]** First, we recalibrate on the new photos, a technique known as AdaBN. The model re-learns what normal looks like in the new place before it labels anything. Known images mistaken for new ones drop from 74 to 78 percent down to 14 to 21 percent.

**[click]** Second, the bridge, a small translator between the new features and the frozen old classifier. The paper only trains it towards a pass-through, and 13 percent error remains. We build it as an exact pass-through, so it starts with zero error.

**[click]** Third, the model labels the new photos itself, so we let it learn only from the fifth it is surest about. On MNIST to USPS, accuracy on known classes rises from 52.8 to 83.4.

**[click]** Fourth, the image network was already trained on millions of photos, so for office photos we keep it fixed. Retraining it on our 900 photos dropped its accuracy on the new things from 72 to about 30.

## Slide 9. Results (30 s)

Read the table left to right. As written, the method sits 3 to 50 points below the paper across all 17 transfers.

**[click]** With our four changes, digits recover: 71.6 against the paper's 72.5 on MNIST to USPS,

**[click]** and 68.3 against 69.4 on USPS to MNIST, both within noise.

**[click]** On SVHN to MNIST we reach 58.2 against 46.4, above the paper, and above it on the new classes too.

**[click]** On office photos, [Office result with our changes: fill in once report Table 3 has it].

**[click]** VisDA with our changes we did not run, and I will come back to that.

## Slide 10. Limitations (25 s)

Four limits.

**[click]** We never ran VisDA with our changes. The cluster gives one GPU for six hours at a time, and ResNet-50 over VisDA with five seeds did not fit what we had left.

**[click]** Accuracy on the new classes is still below the paper, except on SVHN to MNIST.

**[click]** Our changes alter the method, which is why we report the method as written separately.

**[click]** And we did not implement two parts of the paper: Ours-d, which uses real source features, and the two-step upgrade from Amazon to DSLR to Webcam.

## Slide 11. Future work (20 s)

Next, **[click]** run VisDA with the four changes, **[click]** close the new-class gap, **[click]** and find a label-free way to choose the epoch. Four lessons carry over to similar methods: recalibrate on the new data first, start the bridge as an exact pass-through, learn only from confident labels, and check thresholds in high dimensions against the right distribution. Thank you.

## Where each number comes from

"Report" means the team report, `report/` on the `report` branch (PDF of commit `d153099`). All of our results are the mean of 5 seeds at a fixed epoch 15 (VisDA at its last epoch, 10), chosen without target labels.

| Slide | Number | Source |
| ----- | ------ | ------ |
| 3 | 28.6, 62.1, 66.3, 63.0, 70.8 | Paper, Table 3 (Office-31 average, accuracy on new classes). Earlier methods use 5% labelled new photos. |
| 3 | Check marks | Paper, Table 1 |
| 6 | 11 to 26, 3 to 25, about 50 points below | Report, Section 4.2 and Table 3 (baseline, `preset=vanilla`) |
| 6 | New classes 44.2 to 59.0, 9.0 to 39.9, 13.9 to 16.1 against 45.7 to 71.6, 61.8 to 88.4, 55.7 to 56.3 | Report, Table 3, PRIV columns (Office covers both the 10/20 and 20/31 settings) |
| 6 | Bars: 72.5 vs 47.0, 69.4 vs 49.6, 46.4 vs 35.1, 72.2 vs 55.8, 56.4 vs 6.2; 17 transfers | Report, Table 3 (Office bar is A→D in the 10/20 setting) |
| 6 | 10 settings unspecified, 10 vs 11 Office classes, 3σ test keeps almost every point | Report, Table 2 and Section 3.3 |
| 7 | The feedback loop; best epoch between 0 and 12 | Report, Section 5.1, "The model trains on its own mistakes" |
| 7 | 38 to 44% (37.5 to 43.9), 57 to 60%, up to 94%, 10% (10.4) and single-class collapse | Report, Sections 4.2 and 5.1 |
| 7 | 1 to 19 points on Office-31, 24 to 33 on VisDA at the best epoch | Report, Section 5.1, "Our reproduction or the method" |
| 8 | 74 to 78% → 14 to 21% | Report, Section 5.1, cause 1 (Digits M→U) |
| 8 | 13% (24% on VisDA) → 0 | Report, Section 5.1, cause 2, and Section 3.2, change 2 (exact identity at the start) |
| 8 | 52.8 → 83.4 | Report, Section 5.1, cause 3 (Digits M→U, shared accuracy at epoch 30, other changes on) |
| 8 | 27.0 to 33.1 → 72.4, approximately 900 photos | Report, Section 5.1, cause 4 (Office A→D, nearest-guide accuracy on private classes) |
| 9 | Paper columns | Paper, Tables 2 and 3, as listed in report Table 3 |
| 9 | As written: 47.0/54.0, 49.6/59.0, 35.1/44.2, 6.2/16.1 (Real→Syn 7.7/13.9) | Report, Table 3, Baseline columns |
| 9 | With our changes: 71.6/57.5, 68.3/54.4, 58.2/52.6 | Report, Table 3, Improved columns |
| 9 | Office row: as written 57.3/32.7, paper 70.9/70.8 | Mean of the six 10/20 rows of report Table 3. **With our changes: pending, the report's Improved columns for Office are still empty.** |
| 10 | One GPU, six hours, one job at a time | `README.md` line 34, CCDS GPU TC2 QoS `normal`; report Section 4.1 |
| 10 | 57.5, 54.4, 52.6 against 71.6, 68.6, 45.7; Ours-d and A→D→W not implemented | Report, Table 3 and Section 5.3 |
| 11 | Epoch 15 for every task; baseline peaks by epoch 12; the four lessons | Report, Sections 4.1, 5.1 and 6 |
