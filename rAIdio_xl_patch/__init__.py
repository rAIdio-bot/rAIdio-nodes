"""ACE-Step 1.5 XL architecture patch for ComfyUI.

Monkey-patches AceStepConditionGenerationModel to support separate encoder/decoder
dimensions, enabling XL model loading on ComfyUI 0.14.x without upgrading.

Licensed under GPL-3.0 — derivative work of ComfyUI (GPL-3.0).
Source: https://github.com/rAIdio-bot/rAIdio-nodes
"""
import logging
import folder_paths

# ---------------------------------------------------------------------------
# Monkey-patch: ACE-Step 1.5 XL architecture support
#
# ComfyUI 0.14.x uses a single hidden_size for both the decoder (DiT) and
# encoder (lyric/timbre/tokenizer) in AceStepConditionGenerationModel. The XL
# variant has decoder hidden_size=2560 but encoder hidden_size=2048. Upstream
# ComfyUI 0.18+ fixes this by adding encoder_hidden_size, encoder_num_heads,
# and encoder_intermediate_size parameters. We apply the same fix at runtime
# so the base model (hidden_size=2048) continues to work identically.
# ---------------------------------------------------------------------------

def _apply_ace_step_xl_patch():
    """Patch AceStepDiTModel and AceStepConditionGenerationModel to support
    separate encoder/decoder dimensions (required for XL models)."""
    try:
        import torch
        import torch.nn as nn
        import itertools
        import comfy.ldm.ace.ace_step15 as ace
        import comfy.model_management

        # Check if already patched (e.g. by a newer ComfyUI version)
        import inspect
        gen_sig = inspect.signature(ace.AceStepConditionGenerationModel.__init__)
        if "encoder_hidden_size" in gen_sig.parameters:
            logging.info("[rAIdio] ACE-Step XL patch not needed — already supported")
            return

        # --- Patch 1: AceStepDiTModel.__init__ — add condition_dim param ---
        _orig_dit_init = ace.AceStepDiTModel.__init__

        def _patched_dit_init(self, in_channels, hidden_size, num_layers, num_heads,
                              num_kv_heads, head_dim, intermediate_size, patch_size,
                              audio_acoustic_hidden_dim, condition_dim=None,
                              layer_types=None, sliding_window=128, rms_norm_eps=1e-6,
                              dtype=None, device=None, operations=None):
            # Call original init (builds everything with hidden_size)
            _orig_dit_init(self, in_channels, hidden_size, num_layers, num_heads,
                           num_kv_heads, head_dim, intermediate_size, patch_size,
                           audio_acoustic_hidden_dim, layer_types=layer_types,
                           sliding_window=sliding_window, rms_norm_eps=rms_norm_eps,
                           dtype=dtype, device=device, operations=operations)

            # If condition_dim differs from hidden_size, rebuild condition_embedder
            if condition_dim is not None and condition_dim != hidden_size:
                from comfy.ldm.ace.ace_step15 import get_layer_class
                Linear = get_layer_class(operations, "Linear")
                self.condition_embedder = Linear(condition_dim, hidden_size,
                                                 dtype=dtype, device=device)

        ace.AceStepDiTModel.__init__ = _patched_dit_init

        # --- Patch 2: AceStepConditionGenerationModel.__init__ — add encoder params ---
        _orig_gen_init = ace.AceStepConditionGenerationModel.__init__

        def _patched_gen_init(self, in_channels=192, hidden_size=2048,
                              text_hidden_dim=1024, timbre_hidden_dim=64,
                              audio_acoustic_hidden_dim=64, num_dit_layers=24,
                              num_lyric_layers=8, num_timbre_layers=4,
                              num_tokenizer_layers=2, num_heads=16, num_kv_heads=8,
                              head_dim=128, intermediate_size=6144, patch_size=2,
                              pool_window_size=5, rms_norm_eps=1e-06,
                              timestep_mu=-0.4, timestep_sigma=1.0,
                              data_proportion=0.5, sliding_window=128,
                              layer_types=None, fsq_dim=2048,
                              fsq_levels=[8, 8, 8, 5, 5, 5],
                              fsq_input_num_quantizers=1,
                              encoder_hidden_size=2048,
                              encoder_intermediate_size=6144,
                              encoder_num_heads=16,
                              audio_model=None,
                              dtype=None, device=None, operations=None):
            # Build boilerplate state (same as original)
            nn.Module.__init__(self)
            self.dtype = dtype
            self.timestep_mu = timestep_mu
            self.timestep_sigma = timestep_sigma
            self.data_proportion = data_proportion
            self.pool_window_size = pool_window_size

            if layer_types is None:
                layer_types = []
                for i in range(num_dit_layers):
                    layer_types.append("sliding_attention" if i % 2 == 0 else "full_attention")

            # Decoder uses hidden_size (2560 for XL), with condition_dim from encoder
            self.decoder = ace.AceStepDiTModel(
                in_channels, hidden_size, num_dit_layers, num_heads, num_kv_heads,
                head_dim, intermediate_size, patch_size, audio_acoustic_hidden_dim,
                condition_dim=encoder_hidden_size,
                layer_types=layer_types, sliding_window=sliding_window,
                rms_norm_eps=rms_norm_eps,
                dtype=dtype, device=device, operations=operations
            )
            # Encoder, tokenizer, detokenizer use encoder_hidden_size (2048 for XL)
            self.encoder = ace.AceStepConditionEncoder(
                text_hidden_dim, timbre_hidden_dim, encoder_hidden_size,
                num_lyric_layers, num_timbre_layers,
                encoder_num_heads, num_kv_heads, head_dim,
                encoder_intermediate_size, rms_norm_eps,
                dtype=dtype, device=device, operations=operations
            )
            self.tokenizer = ace.AceStepAudioTokenizer(
                audio_acoustic_hidden_dim, encoder_hidden_size, pool_window_size,
                fsq_dim=fsq_dim, fsq_levels=fsq_levels,
                fsq_input_num_quantizers=fsq_input_num_quantizers,
                num_layers=num_tokenizer_layers, head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                dtype=dtype, device=device, operations=operations
            )
            self.detokenizer = ace.AudioTokenDetokenizer(
                encoder_hidden_size, pool_window_size, audio_acoustic_hidden_dim,
                num_layers=2, head_dim=head_dim,
                dtype=dtype, device=device, operations=operations
            )
            self.null_condition_emb = nn.Parameter(
                torch.empty(1, 1, encoder_hidden_size, dtype=dtype, device=device)
            )

        ace.AceStepConditionGenerationModel.__init__ = _patched_gen_init

        logging.info("[rAIdio] ACE-Step XL patch applied — encoder_hidden_size supported")
    except Exception as e:
        logging.warning("[rAIdio] ACE-Step XL patch failed (non-fatal): %s", e)


# Apply patch at import time (before any model loading)
_apply_ace_step_xl_patch()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class RaidioXLLoader:
    """Load ACE-Step 1.5 XL diffusion model with explicit architecture params.

    ComfyUI 0.14.x auto-detection returns default hidden_size=2048 for ACE-Step 1.5,
    which is wrong for the XL variant (hidden_size=2560, 32 layers). This node
    provides the correct XL architecture parameters directly, bypassing detection.

    Requires the XL monkey-patch applied above.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_xl"
    CATEGORY = "loaders"

    # XL architecture (from weight inspection of acestep_v1.5_xl_turbo_bf16.safetensors)
    XL_CONFIG = {
        "audio_model": "ace1.5",
        # Decoder (DiT) dimensions
        "hidden_size": 2560,
        "num_dit_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 9728,
        # Encoder dimensions (separate from decoder for XL)
        "encoder_hidden_size": 2048,
        "encoder_intermediate_size": 6144,
        "encoder_num_heads": 16,
    }

    def load_xl(self, unet_name, weight_dtype):
        import torch
        import comfy.sd
        import comfy.utils
        import comfy.model_management
        import comfy.model_patcher
        import comfy.supported_models

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        logging.info("[RaidioXLLoader] Loading XL model: %s", unet_path)

        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)

        # Build config with explicit XL architecture — skip auto-detection
        model_config = comfy.supported_models.ACEStep15(self.XL_CONFIG.copy())

        parameters = comfy.utils.calculate_parameters(sd)
        weight_dt = comfy.utils.weight_dtype(sd)
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        dtype = model_options.get("dtype", None)
        unet_weight_dtype = list(model_config.supported_inference_dtypes)
        if dtype is None:
            unet_dtype = comfy.model_management.unet_dtype(
                model_params=parameters,
                supported_dtypes=unet_weight_dtype,
                weight_dtype=weight_dt,
            )
        else:
            unet_dtype = dtype

        manual_cast_dtype = comfy.model_management.unet_manual_cast(
            unet_dtype, load_device, model_config.supported_inference_dtypes
        )
        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

        if model_options.get("fp8_optimizations", False):
            model_config.optimizations["fp8"] = True

        model = model_config.get_model(sd, "")
        model_patcher = comfy.model_patcher.CoreModelPatcher(
            model, load_device=load_device, offload_device=offload_device
        )
        if not comfy.model_management.is_device_cpu(offload_device):
            model.to(offload_device)
        model.load_model_weights(sd, "", assign=model_patcher.is_dynamic())

        left_over = sd.keys()
        if len(left_over) > 0:
            logging.info("[RaidioXLLoader] Leftover keys: %s", left_over)

        logging.info("[RaidioXLLoader] XL model loaded successfully")
        return (model_patcher,)


NODE_CLASS_MAPPINGS = {
    "RaidioXLLoader": RaidioXLLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RaidioXLLoader": "rAIdio XL Model Loader",
}
