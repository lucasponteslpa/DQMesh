# TFLite quantized inference example
#
# Based on:
# https://www.tensorflow.org/lite/performance/post_training_integer_quant
# https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor.QuantizationParams

import numpy as np
import cv2
import tensorflow as tf
import os
import cv2
# from depth_model.inference.inference import inference

def center_crop(img, dim):
    """Returns center cropped image

    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped from center
    """
    width, height = img.shape[1], img.shape[0]  #process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]

    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

def dimas_inference(img_path, img_rgb=None, out_path=None, do_crop = False, crop_path=None):
    # Load TFLite model and allocate tensors.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the model file
    model_path = os.path.join(script_dir, 'tf_model', 'dimas_512x512.tflite')
    #model_path = "tf_model/midas_model.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Allocate tensors
    interpreter.allocate_tensors()
    # Print the input and output details of the model
    # print()
    # print("Input details:")
    # print(input_details)
    # print()
    # print("Output details:")
    # print(output_details)
    # print()
    # Convert features to NumPy array
    if img_rgb is None:
        features = cv2.imread(img_path)
    else:
        features = img_rgb
    if do_crop:
        dim_size = features.shape[0] if features.shape[0]<features.shape[1] else features.shape[1]
        features = center_crop(features, (dim_size, dim_size))

    # features = cv2.GaussianBlur(features,(7,7),0)
    # If the expected input type is int8 (quantized model), rescale data
    input_type = input_details[0]['dtype']
    features = cv2.resize(features,(input_details[0]['shape'][1],input_details[0]['shape'][2]), interpolation=cv2.INTER_CUBIC)
    features = features/features.max()
    # mean_pre = np.array([0.406,0.456,0.485])
    # std_pre = np.array([0.225,0.224,0.229])
    # features = (features - mean_pre)/std_pre
    if not crop_path is None:
        cv2.imwrite(crop_path,features)
    np_features = np.array(features)
    if input_type == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        print("Input scale:", input_scale)
        print("Input zero point:", input_zero_point)
        print()
        np_features = (np_features / input_scale) + input_zero_point
        np_features = np.around(np_features)

    # Convert features to NumPy array of expected type
    np_features = np_features.astype(input_type)

    # Add dimension to input sample (TFLite model expects (# samples, data))
    np_features = np.expand_dims(np_features, axis=0)

    # Create input tensor out of raw features
    interpreter.set_tensor(input_details[0]['index'], np_features)
    # breakpoint()
    # Run inference
    interpreter.invoke()

    # output_details[0]['index'] = the index which provides the input
    output = interpreter.get_tensor(output_details[0]['index'])

    # If the output type is int8 (quantized model), rescale data
    output_type = output_details[0]['dtype']
    if output_type == np.int8:
        output_scale, output_zero_point = output_details[0]['quantization']
        print("Raw output scores:", output)
        print("Output scale:", output_scale)
        print("Output zero point:", output_zero_point)
        print()
        output = output_scale * (output.astype(np.float32) - output_zero_point)

    # Print the results of inference
    out_max = output.max()
    out_min = output.min()
    out_image = 255*((output[0]-out_min)/(out_max-out_min))

    if not out_path is None:
        cv2.imwrite(out_path, 255*((output[0]-out_min)/(out_max-out_min)))

    return out_image

def midas_inference(img_path, img_rgb=None, out_path=None, do_crop = False, crop_path=None):
    # Load TFLite model and allocate tensors.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the model file
    model_path = os.path.join(script_dir, 'tf_model', 'midas_model.tflite')
    #model_path = "tf_model/midas_model.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Allocate tensors
    interpreter.allocate_tensors()

    # Print the input and output details of the model
    # print()
    # print("Input details:")
    # print(input_details)
    # print()
    # print("Output details:")
    # print(output_details)
    # print()
    # Convert features to NumPy array
    if img_rgb is None:
        features = cv2.imread(img_path)
    else:
        features = img_rgb
    if do_crop:
        dim_size = features.shape[0] if features.shape[0]<features.shape[1] else features.shape[1]
        features = center_crop(features, (dim_size, dim_size))

    # features = cv2.GaussianBlur(features,(7,7),0)
    # If the expected input type is int8 (quantized model), rescale data
    input_type = input_details[0]['dtype']
    features = cv2.resize(features,(input_details[0]['shape'][1],input_details[0]['shape'][2]), interpolation=cv2.INTER_CUBIC)
    features = features/features.max()
    mean_pre = np.array([0.406,0.456,0.485])
    std_pre = np.array([0.225,0.224,0.229])
    features = (features - mean_pre)/std_pre
    if not crop_path is None:
        cv2.imwrite(crop_path,features)
    np_features = np.array(features)
    if input_type == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        print("Input scale:", input_scale)
        print("Input zero point:", input_zero_point)
        print()
        np_features = (np_features / input_scale) + input_zero_point
        np_features = np.around(np_features)

    # Convert features to NumPy array of expected type
    np_features = np_features.astype(input_type)

    # Add dimension to input sample (TFLite model expects (# samples, data))
    np_features = np.expand_dims(np_features, axis=0)

    # Create input tensor out of raw features
    interpreter.set_tensor(input_details[0]['index'], np_features)

    # Run inference
    interpreter.invoke()

    # output_details[0]['index'] = the index which provides the input
    output = interpreter.get_tensor(output_details[0]['index'])

    # If the output type is int8 (quantized model), rescale data
    output_type = output_details[0]['dtype']
    if output_type == np.int8:
        output_scale, output_zero_point = output_details[0]['quantization']
        print("Raw output scores:", output)
        print("Output scale:", output_scale)
        print("Output zero point:", output_zero_point)
        print()
        output = output_scale * (output.astype(np.float32) - output_zero_point)

    # Print the results of inference
    out_max = output.max()
    out_min = output.min()
    out_image = 255*((output[0]-out_min)/(out_max-out_min))

    if not out_path is None:
        cv2.imwrite(out_path, 255*((output[0]-out_min)/(out_max-out_min)))

    return out_image

if __name__ == "__main__":
    img_name = "img_hdrplus_1"
    img_path = "images/"+img_name+".jpg"
    img_depth = img_name + "_depth.png"
    img_crop = img_name + "_crop.png"
    midas_inference(img_path,img_depth, img_crop, do_crop=True)