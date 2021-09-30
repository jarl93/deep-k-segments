# deep-k-segments
This project is the implementation of a deep clustering algorithm that combines the idea of using segments as representatives and auoencoders to find appropriate representations in the latent space of the autoencoder to cluster high-dimensional data.

# Installation
1. Please install the libraries needed listed in the requirements.txt file.
2. Clone the project by executing the following command:

```bash
git clone https://github.com/jarl93/deep-k-segments.git
```

# Generate synthetic data

The synthetic data will be generated based on different sinusoidal functions (2 by default) parametrized by amplitud, frequency, shift and domain where they are defined, once the functions are defined, some gaussian noise will be added (all this parameters can be specified in the config file). 
Then to generate the training and test data several non-linear and linear transformations will be applied to map the data to a higher-dimensional space. Then the autoencoder will take this higher-dimensional data as input for training and testing.

In order to generate synthetic data run the script generate_synthetic.py with the following command:

```bash
python3 kcurves/generate_synthetic.py ./configs/synthetic_generation.yaml
```

**Note:** All the commands should be executed from the main folder of your project (where you executed the git clone).

The default configuration file "./configs/synthetic_generation.yaml" is a configuration file to set the functions parameters to generate the data. You can create your own file or modify the default one, if you create a new one don't forget to change the last parameter when executing the script.


# Visualize the data generated

In order to visualize the data generated you need to run tensorboard as follows:
```bash
tensorboard --logdir ./data/synthetic/plots
```
Then you have to go to localhost in the indicated port after running the above command.

# Train
In order to do the training run the main script in mode train by executing the following command:

```bash
python3 kcurves train ./configs/synthetic_identity_both.yaml
```

In this case the default configuration file is "./configs/synthetic_identity_both.yaml", but you can create a new one or modify this one.

# Test

In order to do the testing run the main script in mode test by executing the following command:

```bash
python3 kcurves test ./configs/synthetic_identity_both.yaml  
```

In this case the default configuration file is "./configs/synthetic_identity_both.yaml", but you can create a new one or modify this one.


# Visualize the encoder output
After do the training and testing you might want to see how the latent space looks like. In order to visualize the encoder output you need to run tensorboard as follows:

```bash
tensorboard --logdir ./models/synthetic
```

Then you have to go to localhost in the indicated port after running the above command.








