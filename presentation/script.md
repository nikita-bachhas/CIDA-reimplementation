# CIDA reimplementation, 5-minute talk script

Deck: `presentation/slides.html`. Open it in Chrome or Edge, press **F** for full screen and **P** for the presenter window with these notes and a timer.

How to use this script.

- Rehearse it until you can say each slide in your own words. The prof marks down reading from a script.
- **[click]** means press the right arrow once, which reveals the next part of the slide.
- Times are targets. The total is 4:55, which leaves 5 seconds of slack.
- Text in square brackets is for the team to fill in before the talk.

| Slide | Topic                                            | Time | Running |
| ----- | ------------------------------------------------ | ---- | ------- |
| 1     | Title                                            | 5 s  | 0:05    |
| 2     | The problem                                      | 25 s | 0:30    |
| 3     | Earlier work and its limits                      | 35 s | 1:05    |
| 4     | How CIDA works, at home                          | 25 s | 1:30    |
| 5     | How CIDA works, in the new place                 | 25 s | 1:55    |
| 6     | Reimplementation 1, built and tested             | 30 s | 2:25    |
| 7     | Reimplementation 2, why the paper's recipe fails | 35 s | 3:00    |
| 8     | Reimplementation 3, four fixes                   | 40 s | 3:40    |
| 9     | Reimplementation 4, gaps in the paper            | 25 s | 4:05    |
| 10    | Results                                          | 30 s | 4:35    |
| 11    | Take-aways                                       | 20 s | 4:55    |

The reimplementation, slides 6 to 9, takes 130 seconds, the largest share of the talk.

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

## Slide 6. Reimplementation 1, built and tested (30 s)

Now our part, which is the core of this project. There was no public code, so we wrote approximately a thousand lines for the method and 500 for analysis. We also wrote 23 tests and cluster jobs that resume themselves. We added a switch that runs the paper exactly as written. That version fell short on all seventeen transfers. On this chart, the gap runs from 13.6 points on SVHN to MNIST to 50.2 points on VisDA.

## Slide 7. Reimplementation 2, why the paper's recipe fails (35 s)

Small errors start early and then snowball.

**[click]**

Before any learning, the bridge only roughly passes features through, so 52 percent of known images already land on the new side.

**[click]**

Next, the calibration still fits the city, so known images sit closer to the new-thing magnets. Only 37.5 to 43.9 percent of the first labels are right.

**[click]**

Then it keeps training on its own wrong labels, until 57 to 60 percent of known images are called new.

**[click]**

So our fixes target the points where the errors start.

## Slide 8. Reimplementation 3, four fixes (40 s)

We made four fixes.

**[click]**

First, we recalibrate on the new photos, known as AdaBN. Known images wrongly labelled new drop from 74 to 78 percent, down to 14 to 21 percent.

**[click]**

Second, we build the bridge to start as an exact pass-through, which lifts accuracy from 44.8 to 71.9 on MNIST to USPS.

**[click]**

Third, the model learns only from its most confident 20 percent of labels, which lifts accuracy from 55.1 to 71.9.

**[click]**

Fourth, for office photos we keep the pretrained image network fixed. Retraining it on approximately 900 photos dropped the new-thing accuracy of its features from 72.4 to 33.1.

## Slide 9. Reimplementation 4, gaps in the paper (25 s)

We also found rules in the paper that do not hold up. Its test for unusual points uses a cut-off of 9, but in 100 dimensions a typical real point already scores about 100. So almost everything passes, and our stricter test changed nothing either. The paper also contradicts itself on the office classes and leaves 11 settings missing or unclear.

## Slide 10. Results (30 s)

Read the table left to right. As written, the method sits 14 to 50 points below the paper on every task. With our four fixes, Digits recovers: 71.9 against 72.5 on MNIST to USPS, within noise, and 58.8 against 46.4 on SVHN to MNIST, above the paper. Office recovers 19 points to 61.3 on average, matching the paper on two of six transfers. What remains is new things on Office, 36.2 against 70.8. The VisDA run with fixes is still pending.

## Slide 11. Take-aways (20 s)

Three lessons stand out. Run a paper exactly as written first, because it shows where accuracy is lost. Treat unstated details as part of the method. And test every cut-off in high dimensions. Next, we want to close the gap on new things on office photos, finish the runs still in flight, and add the paper's two remaining experiments. Thank you.

---

## Where each number comes from

| Slide | Number                                                                              | Source                                                                                                                                                                                        |
| ----- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3     | 28.6, 62.1, 66.3, 63.0, 70.8                                                        | Paper, Table 3 (Office-31 average, accuracy on new classes). Earlier methods use 5% labelled new photos.                                                                                      |
| 3     | Check marks                                                                         | Paper, Table 1                                                                                                                                                                                |
| 6     | 940 and 410 lines, 23 tests                                                         | Repo count of non-blank, non-comment lines in `cida/`, `train_source.py`, `adapt.py` and `scripts/`, and `tests/`                                                                   |
| 6     | 72.5 vs 43.7, 69.4 vs 47.2, 46.4 vs 32.8, 72.2 vs 45.2, 56.4 vs 6.2; 17 transfers   | Paper, Tables 2 and 3. Report, Table 3 (preset=vanilla, 5 seeds)                                                                                                                              |
| 7     | 52%, 37.5 to 43.9%, 58 to 65%, 57 to 60%, 53 to 94%, single-class collapse          | Report, §4.2 / SPEC.md, vanilla preset                                                                                                                                                       |
| 8     | 74 to 78% and 14 to 21%, 44.8, 55.1, 71.9, 33.1 and 72.4, approximately 900 photos  | SPEC.md, "Our choices"                                                                                                                                                                        |
| 9     | Cut-off 9, typical distance about 100, 100% accepted, 11 vs 10 classes, 11 settings | SPEC.md, "Inconsistencies in the paper". Report, Table 2                                                                                                                                      |
| 10    | Paper columns                                                                       | Paper, Tables 2 and 3                                                                                                                                                                         |
| 10    | As written: 43.7/50.5, 47.2/60.7, 32.8/46.5, 6.2/16.1, 7.7/13.9                     | Report, Table 3 (preset=vanilla, 5 seeds, `runs_vanilla/`)                                                                                                                                  |
| 10    | With fixes: 71.9/55.3, 65.5/51.0, 58.8/44.0, 61.3/36.2                              | SPEC.md, "Vanilla results, Digits" (our main runs) and "Main configuration on Office-31". Office row = mean of the six 10/20 transfers (main 367.8/6 and 217.2/6; vanilla 253.0/6 and 91.8/6) |
| 11    | 36.2 against 70.8                                                                   | Same sources as slide 10, Office averages                                                                                                                                                     |
