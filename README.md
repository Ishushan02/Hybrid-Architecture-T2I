# Hybrid-Architecture-T2I
Trying My own New Architecture for Text To Image Generation

Like I have got quite good Ideas to use on, like
- The Intution from Nested Learning Paper of alpha, beta, gamma, delta and theta waves, infuse this ideology in Transformer Architecture such that it behaves as brain like waves each responsible for different functionality
- Infuse SwiGLU Architecture, Llama paper showed it it didn't give proper reason why it's working but it works better the GeLU or ReLU
- Infuse Rotatory PE in Q and K
- RMS Norm
- Gating Mechanism in Transformer, it's one of the award winning paper in Neurips 2025, I can include that
- Obviously TRM as well, I have included it in my previous Architecture as well
- New Idea of Hyper-Connections, a recent paper from DeepSeek is also a fine work which showed it possibility over Normal single lane channel training.. 
- Other Idea of multiple Text Encoder (Qwen, BGE,  etc) nut I don't have that much memory to load all of them and train, I would use Qwen for this project.

Well, but I don't have that much GPU, to try everything hence will come up with something efficient

- I am thinking to scrape the Diffusion process entirely, but to produce fine refinement image we need the denoising step again and again. So, what to do
you know why not just include the denoising process during the forward propagation. 

- I have other thought of why not try it with VEctorized-VAE (VQVAE).. I mean take in the concept of the CodeBook and utilize it as memory and latent representation ?, wanna give it a try ? let's see
