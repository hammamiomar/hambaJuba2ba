import logging
import torch
from diffusers import (
    StableDiffusionPipeline,
    TCDScheduler,
    AutoencoderTiny,
)

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class HyperSD15Pipeline:
    """Simple HyperSD15 pipeline for fast 1-step generation.

    This is a straightforward implementation focused on speed:
    - HyperSD LoRA for 1-step inference
    - Tiny VAE for 3-5x faster decoding
    - Optimized memory layout
    """

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with config.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.pipe = None
        self.device = config.device

        logger.info(f"Initializing HyperSD15Pipeline on {self.device}")

    def load(self):
        """Load model and apply optimizations.

        This is the expensive part (~1 minute first time, then cached).
        """
        logger.info(f"Loading SD1.5 from {self.config.model_id}")

        # Load base SD1.5 pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.config.get_torch_dtype(),
            safety_checker=None,
        ).to(self.device)

        # Load HyperSD LoRA weights for 1-step inference
        logger.info(f"Loading HyperSD LoRA from {self.config.lora_id}")
        self.pipe.load_lora_weights(
            self.config.lora_id,
            weight_name=self.config.lora_filename,
        )
        self.pipe.fuse_lora()  # Merge LoRA into base model

        # Configure TCD scheduler (required for HyperSD)
        logger.info("Configuring TCD scheduler")
        self.pipe.scheduler = TCDScheduler.from_config(
            self.pipe.scheduler.config,
            timestep_spacing="trailing",
        )

        # Apply optimizations
        self._optimize()

        logger.info("Pipeline loaded and optimized")

    def _optimize(self):
        """Apply performance optimizations."""

        # Disable progress bars
        self.pipe.set_progress_bar_config(disable=True)

        # Tiny VAE (3-5x faster decoding)
        if self.config.use_tiny_vae:
            logger.info("Replacing with Tiny VAE")
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd",
                torch_dtype=self.config.get_torch_dtype(),
            ).to(self.device)

        logger.info("Optimizations applied")

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embedding.

        Args:
            prompt: Text prompt

        Returns:
            Prompt embedding tensor (1, seq_len, dim)
        """
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embeddings = self.pipe.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
        return embeddings

    def generate(
        self,
        latent: torch.Tensor,
        prompt: str = "Hamburger in a sewer, sopping wet, surrounded by teeth",
    ) -> torch.Tensor:
        """Generate a single frame from latent noise.

        Args:
            latent: Input latent tensor (1, 4, H//8, W//8)
            prompt: Text prompt for generation

        Returns:
            Generated image tensor (C, H, W) in range [0, 1]
        """
        # Ensure latent is on correct device
        latent = latent.to(self.device)

        # Generate with 1-step HyperSD
        with torch.no_grad():
            output = self.pipe(
                prompt=prompt,
                latents=latent,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                height=self.config.height,
                width=self.config.width,
                output_type="pt",
            )

        return output.images[0]

    def generate_batch(
        self,
        latents: torch.Tensor,
        prompt: str = "Hamburger in a sewer, sopping wet, surrounded by teeth",
        prompt_embeds: torch.Tensor = None,
    ) -> list[torch.Tensor]:
        """Generate multiple frames in one GPU call.

        Args:
            latents: Batch of latent tensors (batch_size, 4, H//8, W//8)
            prompt: Text prompt for generation (ignored if prompt_embeds provided)
            prompt_embeds: Pre-computed prompt embeddings (batch_size, seq_len, dim)

        Returns:
            List of generated image tensors, each (C, H, W) in range [0, 1]
        """
        latents = latents.to(self.device)
        batch_size = latents.shape[0]

        # Use embeddings if provided, otherwise encode prompt
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(self.device)
            kwargs = {"prompt_embeds": prompt_embeds}
        else:
            prompts = [prompt] * batch_size
            kwargs = {"prompt": prompts}

        with torch.no_grad():
            output = self.pipe(
                latents=latents,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                height=self.config.height,
                width=self.config.width,
                output_type="pt",
                **kwargs,
            )

        return [output.images[i] for i in range(len(output.images))]

    def cleanup(self):
        """Free GPU memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

        logger.info("GPU cache cleared")
