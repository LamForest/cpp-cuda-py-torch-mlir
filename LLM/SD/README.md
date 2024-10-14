
## 1.
The DDPM paper describes a corruption process that adds a small amount of noise for every 'timestep'. Given $x_{t-1}$ for some timestep, we can get the next (slightly more noisy) version $x_t$ with:<br><br>

$q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})$<br><br>


That is, we take $x_{t-1}$, **scale it** by $\sqrt{1 - \beta_t}$ and add noise scaled by $\beta_t$. This $\beta$ is defined for every t according to some schedule, and determines how much noise is added per timestep. Now, we don't necessarily want to do this operation 500 times to get $x_{500}$ so we have another formula to get $x_t$ for any t given $x_0$: <br><br>

$\begin{aligned}
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, {(1 - \bar{\alpha}_t)} \mathbf{I})
\end{aligned}$ where $\bar{\alpha}_t = \prod_{i=1}^T \alpha_i$ and $\alpha_i = 1-\beta_i$<br><br>

这个公式和上面的是等价的吗

The maths notation always looks scary! Luckily the scheduler handles all that for us. We can plot $\sqrt{\bar{\alpha}_t}$ (labelled as `sqrt_alpha_prod`) and $\sqrt{(1 - \bar{\alpha}_t)}$ (labelled as `sqrt_one_minus_alpha_prod`) to view how the input (x) and the noise are scaled and mixed across different timesteps:

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013014837.png)

t=999
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013021319.png)

i=899
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013021346.png)


![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013021414.png)

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013021447.png)

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013021522.png)


![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013021556.png)

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013021725.png)

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013021836.png)

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013021921.png)


![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013022018.png)

由于不是分多次生成的，所以图片没有对应性，但能大概看出从noise到butterfly的转变


## 2. train from scratch MNIST


什么是UNET



![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013135658.png)

```py
class BasicUNet(nn.Module):
    """A minimal UNet implementation."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU() # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x)) # Through the layer and the activation function
            if i < 2: # For all but the third (final) down layer:
              h.append(x) # Storing output for skip connection
              x = self.downscale(x) # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0: # For all except the first up layer
              x = self.upscale(x) # Upscale
              x += h.pop() # Fetching stored output (skip connection)
            x = self.act(l(x)) # Through the layer and the activation function

        return x
```

如果尝试训练一个一步从noisy image到clean image的模型，结果会差：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013140121.png)

和这个naive的训练方式，DDPM采用了：
- 不同训练方式，才用多步噪声训练
- timeStep conditioning，当前step也作为模型输入传进去，转为embedding（具体实现？），模型能知道当前是哪个step，从而更有针对性的去噪
- 预测noise而不是denoise image 没看懂，但目前来看预测noise效果更好
> It turns out there's another subtlety here. We compute the loss across different (randomly chosen) timesteps during training. These different objectives will lead to different 'implicit weighting' of these losses, where predicting the noise puts more weight on lower noise levels. You can pick more complex objectives to change this 'implicit loss weighting'. Or perhaps you choose a noise schedule that will result in more examples at a higher noise level. Perhaps you have the model predict a 'velocity' v which we define as being a combination of both the image and the noise dependent on the noise level (see 'PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS'). Perhaps you have the model predict the noise but then scale the loss by some factor dependent on the amount of noise based on a bit of theory (see 'Perception Prioritized Training of Diffusion Models') or based on experiments trying to see what noise levels are most informative to the model (see 'Elucidating the Design Space of Diffusion-Based Generative Models'). TL;DR: choosing the objective has an effect on model performance, and research in ongoing into what the 'best' option is.

At the moment, predicting the noise (epsilon or eps you'll see in some places) is the favoured approach but over time we will likely see other objectives supported in the library and used in different situations.
- UNET结构变化:
The diffusers UNet2DModel model has a number of improvements over our basic UNet above:

GroupNorm applies group normalization to the inputs of each block
Dropout layers for smoother training
Multiple resnet layers per block (if layers_per_block isn't set to 1)
Attention (usually used only at lower resolution blocks)
Conditioning on the timestep.
Downsampling and upsampling blocks with learnable parameters

Let's create and inspect a UNet2DModel:
```
model = UNet2DModel(
    sample_size=28,           # the target image resolution
    in_channels=1,            # the number of input channels, 3 for RGB images
    out_channels=1,           # the number of output channels
    layers_per_block=2,       # how many ResNet layers to use per UNet block
    block_out_channels=(32, 64, 64), # Roughly matching our basic unet example
    down_block_types=(
        "DownBlock2D",        # a regular ResNet downsampling block
        "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",          # a regular ResNet upsampling block
      ),
)
print(model)
```

```
UNet2DModel(
  (conv_in): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (time_proj): Timesteps()
  (time_embedding): TimestepEmbedding(
    (linear_1): Linear(in_features=32, out_features=128, bias=True)
    (act): SiLU()
    (linear_2): Linear(in_features=128, out_features=128, bias=True)
  )
  (down_blocks): ModuleList(
    (0): DownBlock2D(
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 32, eps=1e-05, affine=True)
          (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=32, bias=True)
          (norm2): GroupNorm(32, 32, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (1): AttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Attention(
          (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
          (to_q): Linear(in_features=64, out_features=64, bias=True)
          (to_k): Linear(in_features=64, out_features=64, bias=True)
          (to_v): Linear(in_features=64, out_features=64, bias=True)
          (to_out): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 32, eps=1e-05, affine=True)
          (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
          (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 64, eps=1e-05, affine=True)
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
          (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (2): AttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Attention(
          (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
          (to_q): Linear(in_features=64, out_features=64, bias=True)
          (to_k): Linear(in_features=64, out_features=64, bias=True)
          (to_v): Linear(in_features=64, out_features=64, bias=True)
          (to_out): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 64, eps=1e-05, affine=True)
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
          (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
    )
  )
  (up_blocks): ModuleList(
    (0): AttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Attention(
          (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
          (to_q): Linear(in_features=64, out_features=64, bias=True)
          (to_k): Linear(in_features=64, out_features=64, bias=True)
          (to_v): Linear(in_features=64, out_features=64, bias=True)
          (to_out): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (resnets): ModuleList(
        (0-2): 3 x ResnetBlock2D(
          (norm1): GroupNorm(32, 128, eps=1e-05, affine=True)
          (conv1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
          (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (1): AttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Attention(
          (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
          (to_q): Linear(in_features=64, out_features=64, bias=True)
          (to_k): Linear(in_features=64, out_features=64, bias=True)
          (to_v): Linear(in_features=64, out_features=64, bias=True)
          (to_out): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 128, eps=1e-05, affine=True)
          (conv1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
          (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ResnetBlock2D(
          (norm1): GroupNorm(32, 96, eps=1e-05, affine=True)
          (conv1): Conv2d(96, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
          (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(96, 64, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (2): UpBlock2D(
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 96, eps=1e-05, affine=True)
          (conv1): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=32, bias=True)
          (norm2): GroupNorm(32, 32, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(96, 32, kernel_size=(1, 1), stride=(1, 1))
        )
        (1-2): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 64, eps=1e-05, affine=True)
          (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=32, bias=True)
          (norm2): GroupNorm(32, 32, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
  )
  (mid_block): UNetMidBlock2D(
    (attentions): ModuleList(
      (0): Attention(
        (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
        (to_q): Linear(in_features=64, out_features=64, bias=True)
        (to_k): Linear(in_features=64, out_features=64, bias=True)
        (to_v): Linear(in_features=64, out_features=64, bias=True)
        (to_out): ModuleList(
          (0): Linear(in_features=64, out_features=64, bias=True)
          (1): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (resnets): ModuleList(
      (0-1): 2 x ResnetBlock2D(
        (norm1): GroupNorm(32, 64, eps=1e-05, affine=True)
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
        (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
      )
    )
  )
  (conv_norm_out): GroupNorm(32, 32, eps=1e-05, affine=True)
  (conv_act): SiLU()
  (conv_out): Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)

```
- Sampling methods

Exactly how we take these steps depends on the sampling method used. We won't go into the theory too deeply, but some key design questions are:

- How large of a step should you take? In other words, what 'noise schedule' should you follow?
- Do you use only the model's current prediction to inform the update step (like DDPM, DDIM and many others)? Do you evaluate the model several times to estimate higher-order gradients for a larger, more accurate step (higher-order methods and some discrete ODE solvers)? Or do you keep a history of past predictions to try and better inform the current update step (linear multi-step and ancestral samplers)?
- Do you add in additional noise (sometimes called churn) to add more stochasticity (randomness) to the sampling process, or do you keep it completely deterministic? Many samplers control this with a parameter (such as 'eta' for DDIM samplers) so that the user can choose.

Research on sampling methods for diffusion models is rapidly evolving, and more and more methods for finding good solutions in fewer steps are being proposed. The brave and curious might find it interesting to browse through the code of the different implementations available in the diffusers library [here](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers) or check out the [docs](https://huggingface.co/docs/diffusers/api/schedulers/overview) which often link to the relevant papers.


## Finetuning

#### Faster Sampling with DDIM

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013143544.png)


pred_original_sample 这个又是什么？
prev_example可以理解

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013145031.png)


pipline: 
- 初始化：DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
- 模型 image_pipe.unet
- 推理：image_pipe().images 真方便，一行就可以拿到图片了
- 替换scheduler并推理
```
image_pipe.scheduler = scheduler
images = image_pipe(num_inference_steps=40).images
```
我理解pipeline负责管理模型 + scheduler。同一个模型可以用DDPM、DDIM等不同的scheduler，训练还是DDPM？


#### Finetune描述

1. 获得随机噪声，获得随机timestep，获取clean image
2. 利用scheduler.add_noise，将noise加到cleanimage，得到指定timestep的noise image
```py
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)
```
3. loss=  MSELoss(unet(noise_image, timestep), noise)
```
        noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]

        # Compare the prediction with the actual noise:
        loss = F.mse_loss(
            noise_pred, noise
        )  # NB - trying to predict noise (eps) not (noisy_ims-clean_ims) or just (clean_ims)

```
4. loss.backword()
```
        # Update the model parameters with the optimizer based on this loss
        loss.backward(loss)
```

5. optmizer.step()

一些注意事项：
1. 建议使用gradient accumulation，增加训练稳定性

推理(DDPM)：
1. prev_example = randn(); timestep = 1000
2. 预测noise：pred_noise = unet(prev_example, timestep)
3. denoise：image = scheduler.step(pred_noise, timestep, prev_example).prev_example
4. timestep -= 1
5. 重复2、3直到timestep == 0


#### CLIP guidance

Embed the text prompt to get a 512-dimensional CLIP embedding of the text
For every step in the diffusion model process:
Make several variants of the predicted denoised image (having multiple variations gives a cleaner loss signal)
For each one, embed the image with CLIP and compare this embedding with the text embedding of the prompt (using a measure called ‘Great Circle Distance Squared’)
Calculate the gradient of this loss with respect to the current noisy x and use this gradient to modify x before updating it with the scheduler.

注意几点：
1. guidan只发生在推理，SD模型的训练是否也有guidance?
2. guidance步骤如下：
    - text_embed = clip_model.text_model(text_prompt)
    - 预训练模型预测噪声：noise_pred = image_pipe.unet(prev_sample, timesteps)
    - 得到clean image分布下的降噪图片：ori_image = image_pipe.scheduler.step(noise_pred, timesteps, prev_sample).pred_original_sample
    - 计算cliplossimg_embed = clip_model.image_model(ori_image)
       guidan_loss = some_loss_func(img_embed, text_embed)
    - 根据cliploss计算prev_sample应该调整的方向：torch.autograd.backward(guidan_loss, prev_sample)
      prev_sample -= prev_sample.grad * some_factor

    所以guidan不会影响预测的噪声，只是调整pre_sample，从而让denoise的结果在clean image分布下更接近text prompt。

3. some_factor: 降噪一开始较小，反正都是乱码，CLIP也起不到作用。

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013161734.png)


## SD

Latent Space Diffusion 

而不是 Image Space Diffusion

text embeddings != token embeddings


 An input prompt is first tokenized (based on a large vocabulary where each word or sub-word is assigned a specific token) and then fed through the CLIP text encoder, producing a **768-dimensional (in the case of SD 1.X) or 1024-dimensional (SD 2.X)** vector for each token. To keep things consistent prompts are always padded/truncated to be 77 tokens long, and so the final representation which we use as conditioning is a tensor of **shape 77x102**4 per prompt.

Text Embedding + time embeding 加入cross attention

 Classifier-free Guidance: 解决不按照text prompt生成图片的问题


 attention slicing : 所有head的QKV不一起计算，而是分开算


 negative prompt 如何实现

#### Pipline components

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013170745.png)

#### VAE

Working with these information-rich 4x64x64 latents is more efficient than working with massive 512px images, allowing for faster diffusion models that take less resources to train and use. The VAE decoding process is not perfect, but it is good enough that the small quality tradeoff is generally worth it.

VAE是怎么训练的？


#### Tokenizer, Embedding



#### UNET
输入从prev_sample、timestep增加了text embedding:
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241013171208.png)




#### Scheduler

