(FOR YC - This is a sample model and highlights the process. We have working models for more complex use-cases such as anpr, face-reco, depth, drl, speech etc)

# Google visual wake word challenge 2019 ( VWW VWWC 2019) submission https://tinyurl.com/wakewordchallenge

(For any details and queries pls contact - amit_mate@hotmail.com)

modelVisualWakeWord.h5 => Keras model file

modelVisualWakeWord.tflite => tensorflow lite size optimized model (post training optimization)

Model:

Model is based on Resnet v1 for n=2 , aka Resnet14. Several modifications, as outlined below have been made to the baseline model to address the given challenge and the specified embedded constraints. A two layer FC network has been used at the output with Dropout to regularize. Additional kernel L2 regularization has been added to the short-cut conv layer filter at the last stage. A sigmoid activation function is used since it is a two class problem and binary cross entropy is used as loss function. The number of filters have been limited to 8 to account for SRAM limitation of 250K. The depth has been limited to 14 (n=2) based on MMAC limitation of 60 MMAC. The model size is ~47K parameters, the disk size of unoptimized tflite  version is 188KB, however when optimized for size, the tflite file (modelVisualWakeWord.tflite) is 63K which is below the 250K flash limitation. The model was developed and trained using Keras python tools and converted using Google tensorflow lite tools. The model outputs a probability (p) of the "Person" class object present in the input image. The probability of "Not Person" class object is implicitly (1-p). An additional comparison ( >0.5) is then used to label output as 1 (>0.5) or 0 (<=0.5).

Model Training and Assumptions:

Model has been trained on coco2014 train dataset as indicated in the challenge. The validation set was chosen from coco2014 val dataset. The input image resolution to the model was fixed at 96x96x3 considering SRAM, MMAC constraints and number of conv filters required for feature extraction . CV2 tools were used to scale and resize the images from train/val set maintaining original aspect ratio. First all images from training and validation sets (including gray) were scaled to 640x640x3 maintaining aspect ratio. The images which had the largest person object less than 0.5% of the total area (640x640x3) were then dropped from the set. Then all images were scaled down to 96x96x3. The 0-255 RGB values were scaled between -1 and 1 by using the formula (x/127.5 -1.0) before training.Random Data Erasing and Random flipping were deployed to augment training data during runtime.

Deep CNN, ConvLSTM, Resnet v2 and ResNexT network architectures were also evaluated, however the Resnet v1 architecture was empirically found to be more accurate and quick to train esp under the embedded constraints. Two class categorical cross entropy (softmax activation) AND focal loss were also considered as alternative loss functions. Two class softmax is mathematically equivalent to single-class sigmoid activation. Focal loss however needs more hyper-parameter tuning, but can yield better results. 2x2 kernels were also evaluated considering compute constraint, however additional depth resulting from smaller kernel use did not contribute much to accuracy.

Model evaluation:

The evaluation was done with both h5 and tflite (size optimized and not optimized) models on cocominival set (excluding the "small" persons as specified in the challenge). The accuracy of the model was consistent at 88.3% and above across all three models. (Pls note that the small (< 0.5% of area) person images as noted in the challenge were dropped from the set for accuracy measurement).

On evaluating the images from cocominival set that were identified incorrectly by the model, several errors were found with coco dataset itself some of which are listed below. If these annotations are fixed, the reported accuracy of the model will be even higher.

1. COCO_val2014_000000057904.jpg => hand not annotated properly 
2. COCO_val2014_000000202154.jpg => not annotated properly, not a person 
3. COCO_val2014_000000511453.jpg => lady on beer bottle not annotated 
4. COCO_val2014_000000538064.jpg => human shaped statue not labelled as person, elsewhere inanimate person shaped objects are
5. COCO_val2014_000000109797.jpg => portrait not annotated properly 
6. COCO_val2014_000000449798.jpg => person shaped toys not annotated, elsewhere it is 
7. COCO_val2014_000000462632.jpg => person on laptop not annotated, elsewhere pics etc are annotated 
8. COCO_val2014_000000524245.jpg => person not annotated

Analysis of errors shows that ~60% of errors (i.e., images misidentified by the model) are images annotated as persons. These can be further classified as:

1. Small visible body parts e.g., just hands or just feet
2. Fully covered body - e.g., gloves+coat+helmet
3. Gray or Low-light images 
4. Small persons , esp in crowd 

The remaining ~40% of errors are images annotated as non-persons, confusion is caused by objects with human like features.These can be classified as

1. Animals - cats, cows, giraffes, pandas
2. Chairs
3. Toys - teddy bears and such
4. Human form hallucination

Future work:

Train using random-cropping (object aware) and ZCA for data augmentation. Focal loss function with hyper-parameter optimization using non-convex optimization techniques. Couldnt try these due to cloud compute limitations. The current model is regularized enough but stops learning around 88% train/val accuracy. Based on error analysis,augment data in the error categories identified above and it should scale to higher accuracies even with the current embedded constraints.
