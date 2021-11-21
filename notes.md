# Questions / Research

### Categorical Cross-entropy vs. Sparse Categorical Cross-entropy
*14 November, 2021*

- When testing out making a custom model-training loop using Keras, the demo 
  project on the Keras website used `SpareCategoricalCrossentropy` for the loss 
  function:  
  https://github.com/keras-team/keras-io/blob/eac0837ef3966017acf807d7e7cb404496eb6d86/guides/writing_a_training_loop_from_scratch.py#L62

- The tensor sizes input into `SpareCategoricalCrossentropy` in the demo project are: 
    - `target` (y-labels): `(64,)`
    - `output` (feed-forward training predictions: `(64, 10)`
  
- This means that `SpareCategoricalCrossentropy` doesn't require same-size 
  tensors to operate on
- When trying to implement a similar custom training loop for this project, the 
loss function used was `CategoricalCrossentropy`
- This failed to work, however, as the `CategoricalCrossentropy` function 
  first compares the shapes of the `target` and `output` tensors, then 
  throws an error in they're different
  
*Look into* `SpareCategoricalCrossentropy` *and*
`CategoricalCrossentropy`, *and see why they have different rules 
for data (i.e. tensor) sizing*

**Update:**
- One difference is that `CategoricalCrossentropy` works with categorical labels (i.e. integer labels coded into binary 
  arrays via one-hot encoding), while `SpareCategoricalCrossentropy` works with integer labels
- This was found after encountering errors stemming from mixing "sparse" and "non-sparse" categrotical cross-entropy 
  functions in the `Keras` API, then later confirmed in the `Keras` documentation:
  https://keras.io/api/metrics/accuracy_metrics/#categoricalaccuracy-class 
