# Training

#### September 25
- All data io and model io finished
- Visualization is left to do
- Might be to optimize the code a bit by meta programming in io files. But leave this until later.
- Seems that ec0 is not training with all the date. But when there are only a few (<= 10), ec0 is adapting well. This is the main problem right now
  - I need a systematic way to track this by tuning the learning rate
  - When training rate is about 1, error decreases on the order of O(10) per epoch (not meta_epoch) (error is about O(500000)).
  - When trainign rate is about 10, error decreases by O(10000) (not meta_epoch) (error is about O(500000)).
  - When trainign rate is about 100, error decreases by O(10000) (not meta_epoch) (error is about O(100000)).
  - **The mistake is found, if batch_size = 500, weyl is returning a (500,) shaped vector while target is (500, 1). The broadcast to (500, 500)**

  #### September 26
  - Experience exploding error rates in training
    - Tried: gradient clipping (use clipnorm key word in adam). Fixed learning rate at 0.001
      - clipnorm = 10 (lr = 0.001), error still gets blow up
      - clipnorm = 1 (lr = 0.001), no blowing up for many metaepochs (13). But each epoch learns by reducing error by at most 0.5
      - clipnorm = 1, learning rate = 0.01, error reduces by 0.5 each time. essentially no training
      - clipnorm = 5, learning rate = 0.01, error reduces by 0.5 each time. essentially no training
      - clipnorm = 10 (lr = 0.001), error still gets blow up but actually learns. Reduce error at O(5) in mse
      - clipnorm = 10 (lr = 0.01), error blow up observed.
- needs to log every epoch, not just meta epochs
- Implemented early stopping: if current loss is N times the previous error, restore previous model
- Now able to implement weyl at the end, so weyl + prediction as output of model ec0
- Train a model for ec_3 over night: ec_3 is not weyl assisted

- ec3 doesn't train at all. Probably too big to fit into memory. current parameter gives a size of 1.7 million parameters
-
