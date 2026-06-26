# Why AI models forget what they just read

Subtitle: I tried to catch context rot in the act, using Gemma-2, LongMemEval, attention heads, and sparse autoencoders.

Alternate titles:

1. Why long-context AI models forget what they just read
2. I tried to find where context rot happens inside a language model
3. The answer was in the prompt. The model still forgot it.
4. Context windows are not memory
5. Watching an AI model lose the plot

---

A strange thing happens when you give a language model a long prompt.

The answer can be right there. Not implied. Not hidden. Literally present in the text.

And the model still gets it wrong.

This is the thing people call context rot. The model technically has room left in its context window, but its ability to retrieve the relevant fact starts degrading as the prompt fills up. If you've used long-context models for research, coding, legal docs, or personal memory, you've probably felt this. The model starts out sharp. Then you keep adding material. At some point it starts missing things it should obviously see.

The usual explanation is behavioral: accuracy goes down as context gets longer. Useful, but unsatisfying. I wanted to know what was happening inside the model when it lost access to a fact.

So I built a small mechanistic experiment around a simple question:

When a model forgets something that is still in its prompt, can we see the retrieval machinery fail?

Short answer: yes, at least in this setup. The internal signal weakens as the context fills, and the point where it weakens predicts the point where the model's answer flips.

The stronger version is tempting: this is a mechanism for context rot. I would not quite say that yet. The evidence is suggestive, not final. But it is a useful case study, and the failure is more concrete than "long prompts are hard."

## The setup

I used Gemma-2-2B-it, an 8k-context open model, and built controlled haystacks from LongMemEval-S.

Each example has:

- one question
- one gold session containing the answer
- a variable number of distractor sessions

The key trick is that I replay the same question at different fill levels:

k = 0, 1, 2, 4, 8, 14 distractor sessions

The gold session stays in the prompt. Its position is randomized and logged. The distractors make the context fuller, but the answer does not disappear.

That matters. If the model fails, it is not because the fact was removed. It is because the model stopped retrieving it.

I kept 140 LongMemEval-S questions that fit the controlled setup: non-abstention, exactly one gold evidence session after trimming around the answer turn when needed, and no preference-rubric questions that my heuristic grader could not score cleanly.

This is not the full LongMemEval benchmark. It is a controlled subset built for mechanistic inspection.

## What happened behaviorally

Accuracy dropped hard as the context filled.

| distractor sessions | 0 | 1 | 2 | 4 | 8 | 14 |
|---:|---:|---:|---:|---:|---:|---:|
| accuracy | 0.714 | 0.671 | 0.579 | 0.586 | 0.521 | 0.343 |

The trend is clear, though not perfectly monotonic. k=4 is a hair better than k=2, so the honest phrasing is "overall decline," not "monotonic decline."

By k=14, accuracy fell from about 71% to 34%, with the gold evidence still present and within the model's 8k window.

This is the basic context-rot behavior.

But behavior alone was not the interesting part.

## Looking inside the model

I tracked two internal signals.

First: retrieval-head attention mass.

Some attention heads behave like copy/retrieval heads. When the model answers a question, these heads often place attention on the evidence span and help move information from the prompt into the generated answer. I identified 20 such heads with a copy-score probe, then measured how much attention they put on the gold span while the model generated its answer.

Second: sparse-autoencoder features.

Gemma Scope provides sparse autoencoders trained on Gemma residual streams. I used them as a microscope for residual activations. The rough idea is that some SAE features become active when the model is recalling the relevant fact. I selected recall-associated features and tracked their activation as the context filled.

Both signals declined with k.

The retrieval-head attention mass on the gold span got weaker as distractors were added. The SAE recall-feature signal weakened too. This happened even though the gold evidence was still in the prompt.

That by itself is already interesting. It suggests that context rot is not just a bad final answer. The model's internal route to the answer is degrading.

## The paired test

The cleanest part of the experiment is the paired within-question design.

For each question, I asked two things:

1. At what fill level does the internal retrieval signal collapse?
2. At what fill level does the model's answer first become wrong?

Among the 100 questions the model answered correctly at k=0, the collapse point predicted the answer-flip point.

For retrieval-head mass:

- Spearman rho = 0.731
- p = 3.6e-9
- n = 48 questions where both collapse and flip were observed

For the SAE feature signal, the correlation was similar: rho = 0.763.

This is the core result. Not just "longer context makes the model worse," but "the per-question internal failure point lines up with the per-question behavioral failure point."

That makes the story much more mechanistic.

## The causal poke

Correlation is cheap. I also wanted to poke the system.

So I tried two interventions.

First, I mean-ablated the 20 retrieval heads at k=0, on questions the model originally answered correctly. Accuracy dropped by 0.82. A matched random-head control dropped by 0.25.

That says these heads are not decorative. Removing their normal contribution hurts retrieval-heavy answering a lot more than ablating random heads.

Second, I tried a feature-mode intervention at high k. For questions that were correct at k=0 but wrong at k=8, I re-injected the collapsed SAE recall features during decoding.

This recovered 0.161 of the flipped answers, with a 95% CI of [0.032, 0.290]. In plain terms: about 5 out of 31.

That is not magic. It is not a full fix. But it is not nothing either. Pushing the internal recall signal back up rescued a small but real fraction of failures.

## What I think this means

The experiment points toward a simple picture:

As the context fills, the model's attention to the relevant evidence weakens. Recall-associated residual features weaken too. When those signals collapse for a specific question, the answer often flips.

So context rot may not be a mysterious property of long prompts. At least here, it looks like a retrieval failure: the answer remains available, but the model's internal machinery stops routing it into the answer.

That has practical implications.

If context windows are not reliable memory, then "just stuff everything into the prompt" is a brittle strategy. Bigger windows help, but they do not guarantee retrieval. A fact can be inside the window and still be functionally unavailable.

For systems that need reliable memory, the answer is probably not just longer context. You need retrieval, compression, routing, recency management, explicit citations, maybe external memory. The model's context window is not a database. It is more like a cluttered desk. The paper can be on the desk and still be buried under junk.

## The caveats

There are several.

This is one model: Gemma-2-2B-it. It is small by current standards.

This is one benchmark subset: 140 controlled LongMemEval-S examples, not the whole benchmark.

The grading is heuristic substring/F1, not a full semantic judge. That matters because all the flip/recovery logic depends on correctness labels.

The SAE feature analysis needs stronger held-out validation. In the current pipeline, recall features are selected by correlation with correctness and then analyzed downstream. That is useful for exploration, but it is too circular to treat as final proof. The next version should split feature discovery and evaluation across disjoint questions.

Attention mass also needs stronger normalization controls. If the context gets longer, absolute attention to a fixed span can drop partly because there are more tokens to attend to. The right next check is gold-span attention enrichment over uniform or position-matched distractor spans.

Gemma-2 also uses a mix of sliding-window and global attention layers. If the gold span falls outside a sliding layer's window, that layer structurally cannot attend to it. The paper logs this, but the next version should model it directly.

So I would phrase the result carefully:

This is not a final theory of context rot. It is a mechanistic case study showing that, in a controlled setting, behavioral retrieval failures line up with measurable internal signal collapse.

Still pretty cool.

## Why I care

A lot of AI product design currently assumes that context is memory.

Upload the documents. Paste the chat history. Dump the repo. Add the logs. Give the model everything.

But context is not memory in the human sense, and it is not storage in the database sense. It is an input the model must actively route through a transformer computation. The more clutter you add, the more chances there are for the relevant thing to stop mattering.

The answer can be present and still not be retrieved.

That distinction is going to matter more as models get larger context windows. A million-token window sounds like memory. It is not automatically memory. It is a bigger room to lose things in.

## Reproducibility

The code is seeded and built around a three-stage SLURM pipeline:

1. run the controlled haystack experiment
2. analyze attention/SAE signals and interventions
3. build the report

The paper and figures are in the repo. The heavy artifacts live under scratch in the original run, so the repo currently contains the writeup and code rather than all intermediate tensors/parquets. If I clean this up for broader release, the next thing I should add is a small summary artifact that reproduces every headline number without requiring the full GPU run.

---

Notes for Substack polish before publishing:

- Add one hero figure: the accuracy/context curve or the collapse-vs-flip scatter.
- Link the GitHub repo once it has README consistency fixes.
- Avoid claiming "monotonic" unless the table changes.
- Avoid saying "proved the mechanism." Say "mechanistic evidence" or "case study."
- If posting before more controls, be upfront about the circular SAE-feature caveat. Reviewers will notice; readers will trust you more if you say it first.
