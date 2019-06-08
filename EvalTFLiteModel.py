#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

def eval_model(interpreter, coco_ds):
  total_seen = 0
  num_correct = 0

  for img, label in coco_ds:
    total_seen += 1
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    predictions = (predictions > 0.5).astype(np.uint8)
    
    if predictions == label.numpy():
      num_correct += 1

    if total_seen % 500 == 0:
        print("Accuracy after %i images: %f" %
              (total_seen, float(num_correct) / float(total_seen)))

  return float(num_correct) / float(total_seen)

#function to print evaluation accuracy stats
def eval_data(x_test,y_test) :
    images, labels = tf.cast(x_test, tf.float32), y_test
    print(images.shape)
    print(labels.shape)
    coco_ds = tf.data.Dataset.from_tensor_slices((images, labels)).batch(1)


    interpreter = tf.lite.Interpreter(model_path="modelVisualWakeWord.tflite")
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    eval_model(interpreter,coco_ds)

