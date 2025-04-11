# Removing Transient Distractors from 3D Gaussian Splat Scenes
Based on minor modifications to the T-3DGS repository by Vadim Pryadilshchikov, Alexander Markin, Artem Komarichev, Ruslan Rakhimov, Peter Wonka, and Evgeny Burnaev.
## Brief Discussion of 3D Gaussian Splatting
3D Gaussian Splatting is a novel rendering technique for synthesizing photorealistic 3D scenes from captured data such as multi-view images or videos.  It attempts to supercede Neural Radiance Fields (NeRFs) by offering significantly faster (often realtime) rendering speed, as well as superior image quality in many cases.
### Introduction to the Rendering Method
Unlike traditional mesh-based (rasterization), voxel-based (rasterization), or NURB-based (raytracing) representations, this approach uses a collection of 3D Gaussian functions (anisotropic ellipsoids) positioned in space to approximate scene geometry and appearance.  Each Gaussian (known henceforth as a "splat") has attributes like position, orientation, scale, opacity, and color. During rendering, these Gaussians are projected onto the image plane and blended in screen space using alpha compositing. Because Gaussians are continuous and differentiable, they can be optimized directly with gradient-based methods, making them highly suitable for view synthesis tasks.
### Gradient Optimization
Gradient optimization methods are used to generate the scene.  Each gaussian splat is parameterized with spatial attributes, and then they are optimized directly using any implementation of gradient descent, minimizing photometric loss between the rendering image and the ground truths (input images or video frames).  Convergence can be sped up in a variety of ways:
	- Instead of using random initialization, initialize the gaussians using sparse structure-from-motion points.
	- Iteratively optimize using SGD (if using smart initialization) or ADAM.
	- Adaptively prune low-opacity or extremely thin gaussians.
Training on objects with sparse images or room-sized scenes with videos typically takes several minutes, whereas larger scenes can take hours or longer, but once optimization is complete, rendering can be performed in real-time (60 fps).
## T-3DGS Method Summary
1. Input frames of a video, along with corresponding camera extrinsics.
	- It is assumed that the camera extrinsics are known due to being recorded at the time of recording the video, but it is also possible to use SLAM to estimate the extrinsics.
2. Train a standard 3D Gaussian Splat scene (ignoring transient distractors).
	- During training, track each gaussian's characteristics temporally and feed the data to a light-weight multi-layer perceptron model to predict if they are transient or static.
	- Then, work backwards from the predicted transient gaussians to determine which pixels in the original images produced them, creating rough masks.
3. Refine the masks.
	- T-3DGS uses an edited version of Meta's SAM model to refine the initial rough masks of transient objects in each of the input frames both spatially and temporally.
	- Masks that are too large are assumed to be outliers and are ignored, but masks that are too small are kept, because outliers will minimally impact the final result (due to so much overlapping data from the input frames) while also serving to potentially reduce some of the "floaters" that gaussian splatted scenes typically contain.
4. Retrain the scene using the new inputs with refined masks, removing transient objects from the scene while also improving convergence.
## Experiment
WORK IN PROGRESS: I've gotten T-3DGS to run its testing routine and work on images I've provided.  The current goal is to apply it to novel data; using occlusive-medium video (underwater footage) or distant video (drone footage).  I have identified training data with labeled extrinsics for underwater images, but am still looking for drone footage with extrinsic data that I can understand enough to convert to the required format.  If time permits, I would love to additionally try to fiddle with SAM to potentially use any of the extended methods that other people have proposed for improving underwater segmentation, but given the time it took to understand the surrounding code, I don't think anything other than very small edits would be realistic.
