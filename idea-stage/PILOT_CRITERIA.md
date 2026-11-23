# Pilot decision criteria (written before the pilot ran)

Pilot: ZTF-BTS, forecast origins 5 d (and 20 d when available), 68% and 95% bands, leave-one-fold-out over the five FLARE
folds, every method of prism/methods.py.

1. Observation (motivation) is CONFIRMED if the re-implemented FLARE bands (Mondrian by predicted class) reach marginal
   coverage within 0.03 of nominal but at least one class (expected SLSN) is under-covered by >= 0.08 at 68%.
2. PRISM is PROMISING if (a) its worst-class coverage is within 0.03 of nominal (>= 0.65 at 68%, >= 0.92 at 95%) and
   (b) its class-averaged interval score is not worse than FLARE's by more than 10%, at 68% and 95%.
3. PRISM is REFUTED (redesign before scaling) if its worst-class coverage is below every baseline's or its mean width
   exceeds twice FLARE's.
4. Any baseline that meets criterion 2 better than PRISM is recorded and becomes the reference to beat.
