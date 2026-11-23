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
| 9     | Results                          | 35 s | 4:15    |
| 10    | Limitations                      | 25 s | 4:40    |
| 11    | Future work                      | 15 s | 4:55    |

The reimplementation, slides 6 to 9, takes 135 seconds, the largest share of the talk. The total is 4:55, which leaves 5 seconds of slack.

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

**[click]** On digits it lands 14 to 29 points below the paper.

**[click]** On office photos, 11 to 54 points below, and on the new classes it is far worse: 1.4 to 26.9 percent, where the paper reports 61.8 to 88.4.

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

**[click]** First, we recalibrate on the new photos, known as AdaBN. Which means we let the model adjusts to the new place first, it re-learns what "normal" looks like there, using the new photos, before it labels anything. Known images mistaken for new ones drop from 74 to 78 percent, down to 14 to 21 percent.

**[click]** Second, we build the bridge to start as an exact pass-through, which is a small translator that turns the new extractor's features into the frozen old classifier's language. This lifts accuracy from 44.8 to 71.9 on MNIST to USPS.

**[click]** Third, the model learns only from its most confident 20 percent of labels, which means it lables the new photos itself, so it learns only from the fifth it is surest about. This lifts accuracy from 55.1 to 71.9.

**[click]** Fourth, the image network turning photos into numbers arrived already trained on millions of photos, so for office photos we leave it untouched — retraining it on our 900 dropped its accuracy on the new things from 72 to 33.

## Slide 9. Results (35 s)

Read the table left to right. As written, the method sits 14 to 50 points below the paper on every task.

**[click]** With our four changes, digits recover: 71.9 against the paper's 72.5 on MNIST to USPS, within noise.

**[click]** 65.5 against 69.4 on USPS to MNIST, a little short.

**[click]** And 58.8 against 46.4 on SVHN to MNIST, above the paper.

**[click]** Office photos recover 19 points to 61.3 overall, but the new classes reach only 36.2 against 70.8.

**[click]** VisDA with our changes we did not run, and I will come back to that.

**[click]** Separately, with the one labelled image chosen by eye and reported at epoch 15, all three digit transfers land above the paper: 83.7 against 72.5, 77.1 against 69.4, and 60.4 against 46.4. A different protocol from the table, so we keep it apart.

## Slide 10. Limitations (25 s)

Four limits.

**[click]** We never ran VisDA with our changes. The cluster gives one GPU for six hours at a time, and ResNet-50 over VisDA with five seeds did not fit what we had left.

**[click]** New classes on office photos are still the gap.

**[click]** And without target labels we cannot pick the best epoch, so we report the last one.

**[click]** Finally, the one labelled image matters more than the paper suggests, and we have not isolated how much.

## Slide 11. Future work (15 s)

Next, **[click]** run VisDA with the four changes, **[click]** close the new-class gap on office photos, **[click]** and find a label-free way to choose the epoch. The take-away: running the paper exactly as written, first, is what made all of this visible. Thank you.

## Where each number comes from

| Slide  | Number                                                                                                                                   | Source                                                                                                                                                                                                                        |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3      | 28.6, 62.1, 66.3, 63.0, 70.8                                                                                                             | Paper, Table 3 (Office-31 average, accuracy on new classes). Earlier methods use 5% labelled new photos.                                                                                                                      |
| 3      | Check marks                                                                                                                              | Paper, Table 1                                                                                                                                                                                                                |
| 6      | 14 to 29, 11 to 54, about 50 points below; new-class ranges 46.5–60.7, 1.4–26.9, 13.9–16.1 against 45.7–71.6, 61.8–88.4, 55.7–56.3 | Nikkita's baseline report, "Why the CIDA baseline underperforms" (2026-11-23), summary table                                                                                                                                  |
| 6      | 72.5 vs 43.7, 69.4 vs 47.2, 46.4 vs 32.8, 72.2 vs 45.2, 56.4 vs 6.2; 17 transfers                                                        | Paper, Tables 2 and 3. Report, Table 3 (preset=vanilla, 5 seeds)                                                                                                                                                              |
| 6      | 11 settings unspecified, 10 vs 11 Office classes, "unusual point" test accepts about 100%                                                | SPEC.md, "Inconsistencies in the paper". Report, Table 2                                                                                                                                                                      |
| 7      | The feedback loop; best epoch between 0 and 12                                                                                           | Nikkita's report, "The core problem"                                                                                                                                                                                          |
| 7      | 38 to 44%, 57 to 60%; up to 94%; 10% and single-class collapse                                                                           | Nikkita's report, "How the failure shows up on each benchmark"                                                                                                                                                                |
| 7      | 1 to 19 points on Office-31, 24 to 33 on VisDA at the best epoch                                                                         | Nikkita's report, "Is it our reproduction or the method?"                                                                                                                                                                     |
| 8      | 74 to 78% and 14 to 21%, 44.8, 55.1, 71.9, 33.1 and 72.4, approximately 900 photos                                                       | SPEC.md, "Our choices"                                                                                                                                                                                                        |
| 8      | +7.9, +18.3, +5.3 overall and 2 to 7 points lower on new things                                                                          | Sourabh's report, matched all-target against confident-only La1 at epoch 15, 5 seeds                                                                                                                                          |
| 9      | 83.7 / 75.5, 77.1 / 71.0, 60.4 / 59.1                                                                                                    | Sourabh's report, chosen one-shot supports, confident-only La1, epoch 15, 5 seeds (83.73 ± 1.83, 77.10 ± 4.65, 60.37 ± 4.07)                                                                                               |
| 8      | 13% on Digits, 24% on VisDA, projection error after pretraining                                                                          | Nikkita's report, "Why the early pseudo-labels are wrong"                                                                                                                                                                     |
| 9      | Paper columns                                                                                                                            | Paper, Tables 2 and 3                                                                                                                                                                                                         |
| 9      | As written: 43.7/50.5, 47.2/60.7, 32.8/46.5, 6.2/16.1, 7.7/13.9                                                                          | Report, Table 3 (preset=vanilla, 5 seeds, `runs_vanilla/`)                                                                                                                                                                  |
| 9      | With our changes: 71.9/55.3, 65.5/51.0, 58.8/44.0, 61.3/36.2                                                                             | SPEC.md, "Vanilla results, Digits" (our main runs, last epoch, random one-shot images) and "Main configuration on Office-31". Office row = mean of the six 10/20 transfers. **Office still to confirm with Siddharth.** |
| 10     | One GPU, six hours, one job at a time                                                                                                    | `README.md` line 34, CCDS GPU TC2 QoS `normal`                                                                                                                                                                            |
| 10     | Peak near epoch 30, drift to epoch 60                                                                                                    | SPEC.md, Office A→D frozen run (ALL 72.2 / PRIV 52.3 at epoch 30, PRIV 48.5 at epoch 60)                                                                                                                                     |
| 10, 11 | 36.2 against 70.8; 26% of new-class images pulled into known classes                                                                     | Same sources as slide 9, Office averages; SPEC.md, A→D confusion at epoch 60                                                                                                                                                 |
