Yes. For matching this PaDIS repo, your pasted default_parameters() should mostly
  mirror train.py plus training.networks.SongUNet.

  Core Mapping

   This Repo                Pasted Version                        PaDIS DDPM++ Default
  ━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   arch='ddpmpp'            mode="ddpmpp"                                       ddpmpp
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   precond='pedm'           should be a separate                                  pedm
                            params.precond
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   model_type='SongUNet'    NCSNpp class implementation                  SongUNet-like
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   model_channels           params.model_channels /                                128
                            original nf
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   channel_mult             params.channel_mult                              [2, 2, 2]
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   channel_mult_emb         implicit                                                 4
                            params.model_channels * 4
                            time embedding
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   num_blocks               params.num_res_blocks                                    4
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   attn_resolutions         params.attn_resolutions                               [16]
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   dropout                  params.dropout                    CLI default 0.13, README
                                                                             uses 0.05
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   embedding_type           params.embedding_type                         "positional"
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   channel_mult_noise       params.channel_mult_noise                                1
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   encoder_type             params.encoder_type                             "standard"
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   decoder_type             params.decoder_type                             "standard"
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   resample_filter          related to fir_kernel /                             [1, 1]
                            resampling setup
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   implicit_mlp             params.implicit_mlp                                  False
  ───────────────────────  ──────────────────────────────  ────────────────────────────
   use_fp16                 params.use_fp16                                      False

  Input/Output Mapping

   This Repo                        Pasted Version             Notes
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━
   img_resolution                   geometry.image_shape[1]    PaDIS uses padded
                                                               resolution in training.
  ───────────────────────────────  ─────────────────────────  ─────────────────────────
   img_channels as network input    params.input_channels      In PaDIS this is image
                                                               channels plus position
                                                               channels.
  ───────────────────────────────  ─────────────────────────  ─────────────────────────
   out_channels                     params.output_channels     Usually original image
                                                               channels, e.g. 1.
  ───────────────────────────────  ─────────────────────────  ─────────────────────────
   label_dim                        class-conditioning         PaDIS usually has
                                    parameter, not             label_dim=0.
                                    conditional
  ───────────────────────────────  ─────────────────────────  ─────────────────────────
   augment_dim                      no clean equivalent in     PaDIS sets
                                    pasted code                augment_dim=9 when
                                                               augmentation is
                                                               enabled.

  For this repo’s patch model, the effective input channels are built in training/
  training_loop.py:102:

  net_input_channels = img_channels + 2

  So for grayscale CT:

  params.input_channels = 1 + 2  # image + x/y position channels
  params.output_channels = 1

  Your pasted line:

  self.input_channels = geometry.image_shape[0] + 2 * params.positional_encoding

  is conceptually close, assuming params.positional_encoding == 1.

  Patch/Training Mapping

   This Repo        Pasted Version                                       PaDIS Default
  ━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   real_p           params.real_p or                                               0.5
                    params.p_largest_patch
  ───────────────  ─────────────────────────────────────  ─────────────────────────────
   padding          params.padding                                                True
  ───────────────  ─────────────────────────────────────  ─────────────────────────────
   pad_width        params.pad_width                                                64
  ───────────────  ─────────────────────────────────────  ─────────────────────────────
   four_channels    params.four_channels                                             1
  ───────────────  ─────────────────────────────────────  ─────────────────────────────
   hash_channels    params.hash_channels                                             1
  ───────────────  ─────────────────────────────────────  ─────────────────────────────
   progressive      params.progressive_training, not                             False
                    original NCSN++ progressive
  ───────────────  ─────────────────────────────────────  ─────────────────────────────
   patch sizes      likely custom param                      [16, 32, 64] when padding
                                                                               is true

  Important: this repo’s progressive is about patch-size curriculum. In your pasted
  NCSN++ code, progressive means progressive output architecture from the original
  NCSN++ implementation. Those are not equivalent.

  Loss/Noise Mapping

   This Repo               Pasted Version                  PaDIS Default
  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━
   Patch_EDMLoss.P_mean    params.P_mean                            -1.2
  ──────────────────────  ───────────────────  ──────────────────────────
   Patch_EDMLoss.P_std     params.P_std                              1.2
  ──────────────────────  ───────────────────  ──────────────────────────
   sigma_data              params.sigma_data                         0.5
  ──────────────────────  ───────────────────  ──────────────────────────
   sigma_min               params.sigma_min       0 for training precond
  ──────────────────────  ───────────────────  ──────────────────────────
   sigma_max               params.sigma_max     inf for training precond

  The biggest behavioral mismatch is here: this repo does not use a discrete
  self.sigmas buffer for the default pedm training path. It samples continuous EDM
  sigmas:

  sigma = exp(randn * P_std + P_mean)

  and then Patch_EDMPrecond uses:

  c_noise = sigma.log() / 4

  So this pasted code is not equivalent if it does:

  self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(params)))
  used_sigmas = self.sigmas[time_cond.long()]
  temb = get_timestep_embedding(timesteps, params.model_channels)

  To match this repo, your forward should accept continuous sigma, not integer
  time_cond, and for DDPM++/PaDIS-style patch EDM it should embed:

  noise_labels = sigma.log() / 4
  temb = positional_embedding(noise_labels, model_channels)

  Suggested DDPM++ Defaults

  model_params.embedding_type = "positional"
  model_params.encoder_type = "standard"
  model_params.decoder_type = "standard"
  model_params.channel_mult_noise = 1
  model_params.resample_filter = [1, 1]
  model_params.model_channels = 128
  model_params.channel_mult = [2, 2, 2]
  model_params.num_res_blocks = 4
  model_params.attn_resolutions = [16]
  model_params.dropout = 0.13  # or 0.05 to match README command
  model_params.precond = "pedm"
  model_params.P_mean = -1.2
  model_params.P_std = 1.2
  model_params.sigma_data = 0.5
  model_params.padding = True
  model_params.pad_width = 64
  model_params.real_p = 0.5
  model_params.four_channels = 1
  model_params.hash_channels = 1
  model_params.implicit_mlp = False
  model_params.use_fp16 = False

  The architecture parameter names map cleanly, but the sigma handling does not.
  Matching this repo requires using the PaDIS/EDM preconditioning path, not the
  original score-SDE discrete sigmas lookup path.
