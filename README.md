## AudioCommons Timbral Models for Tensorflow
The timbral models were devleoped by the [Institute of Sound Recording (IoSR)](http://www.iosr.uk/AudioCommons/) at the University of Surrey, and was completed as part of the [AudioCommons project](https://www.audiocommons.org).  


The current distribution contains python scripts for predicting eight timbral characteristics: _hardness_, _depth_, _brightness_, _roughness_, _warmth_, _sharpness_, _booming_, and _reverberation_.

More detailed explanations of how the models function can be found in Deliverable D5.8: Release of timbral characterisation tools for semantically annotating non-musical content, available: http://www.audiocommons.org/materials/

These models were converted (as much as possible) for Tensorflow in order to guarantee gradient propagation for use in deep learning models.

## Using the models and calculating attributes
The models are formatted in a python package that can be simply imported into a Python script.

Here is an example with the brightness model.

```
import timbral_models
from timbral_models import Timbral_Brightness

# Tensorflow-compatible models expect Tensor of shape [batch, length]. There is no automatic stereo collapsing.
fs, audio_samples = .... 

results = Timbral_Brightness.tf_timbral_brightness(audio_samples, fs)
```

## Model output

The *hardness*, *depth*, *brightness*, *roughness*, *warmth*, *sharpness*, and *booming* are regression based models, trained on subjective ratings ranging from 0 to 100.  However, the output may be beyond these ranges.   
The `clip_output` optional parameter can be  used to contrain the outputs between 0 and 100.
```
timbre = timbral_models.timbral_extractor(fname, clip_output=True)
```   
For additional optional parameters, please see Deliverable D5.8 of the AudioCommons project.

For additional optional parameters, please see Deliverable D5.8.

The _reverb_ attribute is a classification model, returning 1 or 0, indicating if the file "sounds reverberant" or "does not sound reverberant" respectively.

## Version History

Version 0.0 : original release. 

## Citation
```
@proceedings{lavault_antoine_2022_6573361,
  title        = {{Stylewavegan: Style-Based Synthesis of Drum Sounds 
                   With Extensive Controls Using Generative
                   Adversarial Networks}},
  year         = 2022,
  publisher    = {Zenodo},
  month        = jun,
  doi          = {10.5281/zenodo.6573361},
  url          = {https://doi.org/10.5281/zenodo.6573361}
}
```

## Citation

For refencing these models, please reference "StyleWaveGAN: Style-based synthesis of drum sounds with extensive
	controls using generative adversarial networks"

## Advancement

### Booming
- [x] Gradients
- [x] Batch compatibility

### Brightness
- [x] Filtering
- [x] Spectrograms
- [x] Gradients
- [x] Batch compatibility
- [x] Optimization

### Depth
- [x] Gradients
- [x] Batch compatibility
- [ ] Onset decay calculation optimization


### Hardness
- [ ] Gradients
- [ ] Batch compatibility


### Roughness
- [ ] Gradients
- [ ] Batch compatibility


### Sharpness
- [x] Gradients
- [x] Batch compatibility

### Warmth
- [x] Gradients
- [x] Batch compatibility

