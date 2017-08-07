# keras-experiments
simple convenience methods to quickly turn downloaded images into your custom classifier via Keras, VGG16 

Features

* Automatic labeling and jpg conversion of a directory of downloaded images
* Center cropping and resizing to 224 x 224 (VGG16 input size)
* Hough circle detection to augment your training w/ noiseless images (if your domain is circular)
* Random Train/Test creation
* Augmentation using Keras ImageDataGenerator
* Extracting and saving VGG16 embeddings, training classifier on your images, test

Keras w/ Theano backend

# Simple Example - Pigeon Classifier

![barb](barb_7.jpg?raw=true "barb")![frillback](frillback_33.jpg?raw=true "frillback")
![pouter](pouter_8.jpg?raw=true "pouter")![carneau](Carneau%20Pigeon.jpg?raw=true "carneau")

See Example.py for simple example on 4-way pigeon classifier that gets over 90% accuracy in minutes

1. Download VGG16
2. Download the pigeons
3. Change paths for VGG16 and pigeons in example.py and run

```python
vgg_weights_path = "C:\\Users\\Big Pigeon\\Downloads\\vgg16_weights.h5"
root_dir = "C:\\Users\\Big Pigeon\\Documents\\Python Scripts\\pigeons"

#adds carneau labels to labeled image dir, creates train/test sets, crop/resize all to 224 by 224
def create_training_test(root_dir):
    all_dir = os.path.join(root_dir, "all")
    hip.create_labels(os.path.join(root_dir, "carneau"), all_dir, "carneau")
    hip.create_train_test(root_dir, all_dir, ["barb", "carneau", "frillback", "pouter"], 0.2)
    raw_train_dir = os.path.join(root_dir, "raw_train")
    raw_test_dir = os.path.join(root_dir, "raw_test")
    hip.crop_resize_to_squares(raw_train_dir, root_dir) #saves to "square224px"
    hip.crop_resize_to_squares(raw_test_dir, root_dir, 320) #subsequent rotation step will resize to [224 x 224]
    test_320px_dir = os.path.join(root_dir, "square320px")
    hip.perform_rotations(90, test_320px_dir, root_dir)

def experiment_harness(root_dir, augmentations, epochs):
    to_augment_name = "square224px"
    augmentation_desc = "imagedatagen_" + to_augment_name + "_" + str(augmentations)
    train_dir = os.path.join(root_dir, augmentation_desc)
    train_pkl = os.path.join(root_dir, to_augment_name + ".pkl")
    test_dir = os.path.join(root_dir, "90_rotations")
    test_dirs = [[test_dir, os.path.join(root_dir, "test.pkl")]]
    m.dataGen(os.path.join(root_dir, to_augment_name), train_dir, augmentations)
    computeAllEmbeddings(train_dir, test_dirs, train_pkl)
    weights_path = os.path.join(root_dir, augmentation_desc + "model")
    m.train_top_model(train_pkl, [test_dirs[0][1]], weights_path, train_dir, epochs)
    
create_training_test(root_dir)
experiment_harness(root_dir, 40, 5)
```

Epoch 498/500
5363/5363 [==============================] - 0s - loss: 7.5461e-04 - acc: 0.9996 - val_loss: 1.2522e-07 - val_acc: 1.0000
Epoch 499/500
5363/5363 [==============================] - 0s - loss: 0.0011 - acc: 0.9998 - val_loss: 0.0107 - val_acc: 0.9993
Epoch 500/500
5363/5363 [==============================] - 0s - loss: 0.0054 - acc: 0.9996 - val_loss: 0.0107 - val_acc: 0.9993
4352/4410 [============================>.] - ETA: 0s
barb  accuracy:  0.907070707070707
carneau  accuracy:  0.8555555555555555
frillback  accuracy:  0.976984126984127
pouter  accuracy:  0.8944444444444445
accuracy:  0.9113378684807256

Not bad at all, especially given the presence of this tricky pigeon - it's considered a frillback but it also looks like a pouter.

![frillback](frillback_26.jpg?raw=true "frillback pouting")

# A Second Experiment - Bottle Caps

## Develop a classifier for bottle caps using a dozen images

I was inspired to create a bottle cap classifier using the Keras framework and VGG16 after reading the article https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html 
To obtain a good classifier that can distinguish between 25 different bottle cap designs I take pictures of caps on a variety of backgrounds and lighting conditions. My training and test sets contain 8-13 and 4-6 original images per cap. To enlarge and perturb the training I use the Keras ImageDataGenerator to rotate, slightly shear and zoom each image 40 times. To verify rotational invariance of the model without distorting the test images I expand the test set by rotating each image 90 times.

## Effects of Including/Excluding Background Noise in Training

[SPOILER ALERT: Including some training images with backgrounds cropped out improved accuracy]

To test the effects of background noise in training I created three training sets and two test sets.

TRAINING
* Background Noise
* No Noise (cropped outside the cap)
* Noise + No Noise

TEST
* Original
* Cropped Around the Cap

The original images contain significant background noise. The cropped images contain no background noise. I use opencv2.HoughCircles to automatically locate the cap in each image and set the remaining pixels to the VGG16 RGB mean values. This circle detection algorithm works very well despite the noise except for four images which I simply throw out. 

![Hough Circle Detection](hough2crop.png?raw=true "Hough Circle Detection")

Since the combined set contains twice the caps as the others, I perturb each of those images 20 times instead of 40. Now I have three training sets of ~10000 images.
For each training and test image I compute the VGG16 embeddings and save to a file. I then take the embeddings from the training set to train a small fully connected model:


| | noise | no-noise |
| noise | 94.0% | 87.3% |
| no-noise |87.9% | 95.8% |
| combined | 96.1% | 96.6% |

## Comparing each Model against the Test Sets

The noisy model is the only model that performs worse on the no-noise test set. This seems like a counterintuitive result for any model as it seems it would be easier to correctly identify test images with no noise. Perhaps some of the identifying edge data was lost in the HoughCircles cropping.  Additionally, the signals from the relatively large cropped-out portion of the no-noise images could disproportionately favor certain classes.

The no-noise model performs better against test images without background noise. It only predicted 3% of Hofbrau caps on a background correctly. The model may have incorrectly identified features in the background noise that resembled those of other cap classes.

## Comparing Models

The noisy model performs better than the no-noise model against the noisy test set. The no-noise model is likely incorrectly associating features in the background as belonging to caps - it did not train at all to ignore noise. 

The no-noise model performs better than the noisy model against the no-noise test set. This could be due to the noisy model slightly overfitting on background features present in the handful of original images. Against a test set containing random background noise the disadvantage of this slight overfitting effect is outweighed by the advantage that it ignores more noise in the test. Only in a test set without noise does this overfitting become apparent.

## Conclusion

The combined model performs better than the other models against both test sets. A few different backgrounds in the training images of each class leads to a model that is good at ignoring noise. Introducing images with cropped out backgrounds then helps reduce the slight overfitting induced by the small set of backgrounds, without reducing the effectiveness of classification through noise.
