# Handwriting generation!

Handwriting generation problem involves 2 sequences: a sequence of text (input) and a sequence of points related to the position of a pen (output). This is an implementation of the paper by [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf) by Alex Graves.

### Dataset Description:

- `data.npy`contains 6000 sequences of points that correspond to handwritten sentences. Each handwritten sentence is represented as a 2D array with T rows and 3 columns. T is the number of timesteps. The first column represents whether to interrupt the current stroke (i.e. when the pen is lifted off the paper). The second and third columns represent the relative coordinates of the new point with respect to the last point.
- `sentences.txt` contains the corresponding text sentences.

#### Tasks :

- Handwriting synthesis
  - Unconditional generation
  - Conditional generation 
- Handwriting recognition

## References:
 [1] https://arxiv.org/abs/1308.0850
