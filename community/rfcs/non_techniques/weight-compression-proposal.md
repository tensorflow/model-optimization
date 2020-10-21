# Keras Weight Compression API

## Objective
Build a Keras-based API and set of guidelines that help people take a [compression](https://en.wikipedia.org/wiki/Data_compression)  algorithm, scale its research to a variety of ML tasks, and productionize it. Productionisation means creating a usable and maintainable API that produces models that are deployable to TensorFlow and TensorFlow Lite and useful to a number of products.

The algorithms benefit serving by reducing storage space, peak memory usage, and network usage. Depending on the compression rate, the computational complexity of the algorithm, and the device, the latency may either increase or decrease.

This will not enable arbitrary compression algorithms and methods of application. This

* Enables algorithms that optimize the weights of a model but not the activations, which includes all [traditional lossless compression](https://en.wikipedia.org/wiki/Lossless_compression) algorithms. This excludes what’s  traditionally called [quantization-aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training) which optimizes the activations.
   * This choice focuses on efforts that would support techniques from groups who have expressed interest in creating Keras-based APIs (e.g. [Entropy Penalized Reparameterization](https://arxiv.org/abs/1906.06624), WEST). Supporting activations would complicate the API, when full-integer quantization is the only “established” algorithm that operates on activations.
* Enables applying algorithms both during-training and post-training.
   * A reason to support post-training is that it’s the best way of applying all lossless compression algorithms, which would result in zero accuracy degradation.
* Enables decompressing the weights either before inference or during inference.
* Excludes algorithms that modify the output shape of a layer (modifying the batch dimension only is okay). So while it enables the variant of structured pruning that improves inference speed, it excludes the variant of structured pruning that improves training speed.
    * This choice was primarily for scoping. In Keras today, we couldn’t support Reshape and we haven’t considered enough how to expose this cleanly for other layers (doc)

## Motivation
Today, many compression researchers fork and modify model and layer code directly. For initial training research for a small number of architectures, this would be the simplest thing to do today, given the maximal flexibility on top of existing TF Core and Keras APIs. It’s not too bad since for weight optimization, there are only a few layers to consider (Dense, LSTM, Conv, and Embedding) for broad model coverage.

This approach restricts the techniques to researchers and advanced ML engineers and it gets trickier beyond initial research, in the following three stages in order:

1. Scaling experiments to several architectures and tasks. Some researchers complain that one of the worst aspects of ML research is having to run and tune experiments. Related to this, It would be painful to manually modify every architecture and potentially maintain these forks.

    a. To focus the researcher on the algorithm, ideally the algorithm’s implementation and the way to apply it should be separate from the base architecture’s implementation. The way to apply the algorithm should work seamlessly with different tasks under Official Models.
Modularity is also important for productionisation, since not all users care about compression (e.g. in Official Models).

2. Translating the results into a production environment.

   a. Many compression researchers have achieved Nx compression with Y% accuracy loss, but have not checked how that runs in practice, including measuring latency. Those that try would run into various issues. With their approaches, they may observe that after compiler optimization, the models are no longer compressed or as compressed, or that decompression during inference increases latency for models with batchnorm. Infrastructure teams are in the best place to solve or create documentation to advise or inform on these issues.

3. Creating a API for model developers to use for adoption of research

   a. The above forking approach means that there isn’t actually a library for other model developers to apply the techniques to different architectures. The APIs in this project will make this much easier.

Making the above easier would encourage researchers and developers to use TF more and work to provide better compression benefits. Even with the forking approach, a few well-tested examples from a library-based approach can be good references.

## Background
TFMOT has released a set of Keras training APIs to make it easy for model developers to apply various model optimization techniques. This project is an extension of that work. Beyond creating an API for others to create techniques, the main difference is that this would additionally allow for decompressing the model during inference.

For decompression during inference, this would reduce the peak memory usage, which can in turn reduce latency and power usage, especially if the resulting model can now fit in a faster and smaller type of memory. In some cases, this can enable using base models that couldn’t fit all before.
