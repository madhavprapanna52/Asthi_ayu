#Bone age prediction Machine learnning project 
Aim 
> Makes bone age predictions through x-ray images

Warnning : I honestly admit that i used ai but with understanding to some extent , I would refine it to credibility

Tools used 
1. Special thanks to chatgpt, perplexity for spending 20 + hours straight in one shot to complete it
2. Tensorflow for machine learnning
3. Heavy use of pretrained feature extraction models to take features out of x-ray images

High level structure 
Model craft 
  Data-set --> Preprocessing_Unit {Making crops, contrast enhancements, transformations of image pixels}
                                              \\\
                                              \/
  Feature_extraction_unit {Extracting features from pretrained models, certain locations of image tracing to take features like total bone area, special points maping vectoric quantity}
  Model_trainning {Simple linear model used with help of keras high level library to lift heavy things and train}

Future upgrade aim 
  1. Using advance ml models for making it more robust to be used in medical feild itself
  2. Advancing image preprcessing engine for greate intellegence of editing
  3. Advancing feature extraction unit to make robust abstraction for good trainning
