"""
Advanced segmentation models for the framework.
This module provides implementations of:
- SegFormer: Hierarchical transformer with MLP decoder
- Mask DINO: DETR-based unified detection/segmentation
- SCTNet: Self-Correcting Transformer Network  
- Enhanced DeepLab variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# Note: Using custom implementation instead of transformers library for compatibility

class SegFormerModel(nn.Module):
    """
    Fixed SegFormer-inspired model with proper per-pixel logits output.
    Compatible with CrossEntropyLoss and your existing train/test/inference logic.
    """

    def __init__(self, num_classes, model_name="segformer-b0", encoder_weights="imagenet"):
        super().__init__()

        self.num_classes = num_classes

        # Patch embedding layers for hierarchical features
        self.patch_embed1 = nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3)
        self.patch_embed2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.patch_embed3 = nn.Conv2d(64, 160, kernel_size=3, stride=2, padding=1)
        self.patch_embed4 = nn.Conv2d(160, 256, kernel_size=3, stride=2, padding=1)

        # Transformer encoder blocks
        self.transformer1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(32, 4, 128, dropout=0.1, batch_first=True), 2)
        self.transformer2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(64, 4, 256, dropout=0.1, batch_first=True), 2)
        self.transformer3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(160, 8, 640, dropout=0.1, batch_first=True), 2)
        self.transformer4 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(256, 8, 1024, dropout=0.1, batch_first=True), 2)

        # Decoder: project all transformer outputs to same dimension
        self.proj_c1 = nn.Conv2d(32, 256, kernel_size=1)
        self.proj_c2 = nn.Conv2d(64, 256, kernel_size=1)
        self.proj_c3 = nn.Conv2d(160, 256, kernel_size=1)
        self.proj_c4 = nn.Conv2d(256, 256, kernel_size=1)

        # Fuse and upsample features
        self.fuse = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Final segmentation head
        self.seg_head = nn.Conv2d(256, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass producing logits (B, num_classes, H, W)."""
        B, C, H, W = x.shape

        # Normalization (ImageNet-style)
        if x.max() <= 1.0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
            x = (x - mean) / std

        # Encoder stages
        c1 = self.patch_embed1(x)
        c1_flat = c1.flatten(2).transpose(1, 2)
        c1 = self.transformer1(c1_flat).transpose(1, 2).reshape(B, 32, c1.shape[2], c1.shape[3])

        c2 = self.patch_embed2(c1)
        c2_flat = c2.flatten(2).transpose(1, 2)
        c2 = self.transformer2(c2_flat).transpose(1, 2).reshape(B, 64, c2.shape[2], c2.shape[3])

        c3 = self.patch_embed3(c2)
        c3_flat = c3.flatten(2).transpose(1, 2)
        c3 = self.transformer3(c3_flat).transpose(1, 2).reshape(B, 160, c3.shape[2], c3.shape[3])

        c4 = self.patch_embed4(c3)
        c4_flat = c4.flatten(2).transpose(1, 2)
        c4 = self.transformer4(c4_flat).transpose(1, 2).reshape(B, 256, c4.shape[2], c4.shape[3])

        # Decoder
        c1_up = F.interpolate(self.proj_c1(c1), size=(H//4, W//4), mode='bilinear', align_corners=False)
        c2_up = F.interpolate(self.proj_c2(c2), size=(H//4, W//4), mode='bilinear', align_corners=False)
        c3_up = F.interpolate(self.proj_c3(c3), size=(H//4, W//4), mode='bilinear', align_corners=False)
        c4_up = F.interpolate(self.proj_c4(c4), size=(H//4, W//4), mode='bilinear', align_corners=False)

        fused = torch.cat([c1_up, c2_up, c3_up, c4_up], dim=1)
        fused = self.fuse(fused)

        # Final per-pixel logits
        logits = self.seg_head(F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False))
        return logits  # shape: (B, num_classes, H, W)

    def predict(self, x):
        """Inference wrapper."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class SimpleViTSegmentation(nn.Module):
    """
    Simple Vision Transformer for segmentation using patch embeddings.
    """
    
    def __init__(self, num_classes, image_size=512, patch_size=16, embed_dim=768, 
                 num_heads=12, num_layers=12, encoder_weights="imagenet"):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Segmentation head
        x = self.seg_head(x)  # (B, num_patches, num_classes)
        
        # Reshape to 2D segmentation map
        patches_per_side = self.image_size // self.patch_size
        x = x.transpose(1, 2).reshape(B, self.num_classes, patches_per_side, patches_per_side)
        
        # Upsample to original resolution
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x
    
    def predict(self, x):
        """Prediction method compatible with test pipeline."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

class MaskDINOModel(nn.Module):
    """
    Fixed & memory-friendly Mask DINO-like module for semantic segmentation.
    Produces (B, num_classes, H, W) logits — compatible with CrossEntropyLoss.

    Notes:
    - This version is intentionally smaller than a full MaskDINO to fit on a 16 GB GPU.
    - It fuses a lightweight transformer global context with a CNN pixel decoder.
    - Feel free to increase hidden_dim / num_queries / decoder layers if you have more VRAM.
    """
    def __init__(self, num_classes, encoder_weights="imagenet"):
        super().__init__()

        self.num_classes = num_classes

        # Memory-conscious defaults for RTX 5080 (16 GB)
        self.hidden_dim = 128      # was 256 -> lower for VRAM
        self.num_queries = 50      # was 100 -> reduce queries
        transformer_decoder_layers = 3

        # CNN backbone (small ResNet-like stem)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
        )

        # Pixel decoder (lightweight upsampling path)
        self.pixel_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Small projection to transformer hidden dim
        self.input_proj = nn.Conv2d(512, self.hidden_dim, kernel_size=1)

        # Transformer decoder (uses queries to extract global context)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=8 if self.hidden_dim >= 128 else 4,
            dim_feedforward=max(512, self.hidden_dim * 4),
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_decoder_layers)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

        # Fusion block: combine pixel decoder features (64 channels) + transformer context (hidden_dim)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim + 64, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Final segmentation head -> logits per class
        self.seg_head = nn.Conv2d(256, num_classes, kernel_size=1)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(1, blocks):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: (B, 3, H, W) in either [0,1] or [0,255].
        returns: logits (B, num_classes, H, W)
        """
        B, C, H, W = x.shape

        # Normalize if input is 0..1
        if x.max() <= 1.0:
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std

        # Backbone features (very downsampled)
        features = self.backbone(x)             # (B, 512, Hf, Wf)

        # Transformer global context
        proj = self.input_proj(features)        # (B, hidden_dim, Hf, Wf)
        memory = proj.flatten(2).transpose(1, 2)  # (B, Hf*Wf, hidden_dim)
        query = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, Q, hidden_dim)
        decoder_out = self.transformer_decoder(query, memory)  # (B, Q, hidden_dim)

        # Aggregate query outputs into a spatial global context
        # Option A: mean pooling over queries (cheap, stable)
        global_context = decoder_out.mean(dim=1)            # (B, hidden_dim)
        global_context = global_context.view(B, self.hidden_dim, 1, 1)
        # expand to feature spatial size
        global_context = global_context.expand(-1, -1, features.shape[2], features.shape[3])  # (B, hidden, Hf, Wf)

        # Local pixel features via pixel decoder
        local_features = self.pixel_decoder(features)       # (B, 64, Hf_up, Wf_up)

        # Upsample global context to match local_features spatial size
        global_up = F.interpolate(global_context, size=local_features.shape[-2:], mode='bilinear', align_corners=False)

        # Fuse along channel dimension
        fused = torch.cat([local_features, global_up], dim=1)  # (B, 64+hidden_dim, Hf_up, Wf_up)
        fused = self.fusion_conv(fused)                       # (B, 256, Hf_up, Wf_up)

        # Final logits upsampled to original image size
        logits = self.seg_head(F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False))  # (B, num_classes, H, W)
        return logits

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class SCTNetModel(nn.Module):
    """
    Fixed Self-Correcting Transformer Network.
    Returns logits of shape (B, num_classes, H, W).
    """

    def __init__(self, num_classes, encoder_weights="imagenet"):
        super().__init__()

        self.num_classes = num_classes
        self.num_correction_rounds = 3

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
        )

        # Transformer refinement
        self.correction_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(256, 8, 1024, dropout=0.1, batch_first=True), 4
        )

        # Decoder (base segmentation)
        self.initial_head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Self-correction modules
        self.error_detector = nn.Sequential(
            nn.Conv2d(64 + 256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

        self.corrector = nn.Sequential(
            nn.Conv2d(64 + 256 + 1, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final segmentation head
        self.seg_head = nn.Conv2d(64, num_classes, 1)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(1, blocks):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        # Normalize input
        if x.max() <= 1.0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
            x = (x - mean) / std

        # Extract and refine features
        features = self.feature_extractor(x)
        B_feat, C_feat, H_feat, W_feat = features.shape
        feat_flat = features.flatten(2).transpose(1, 2)
        refined_features = self.correction_transformer(feat_flat)
        refined_features = refined_features.transpose(1, 2).reshape(B_feat, C_feat, H_feat, W_feat)

        # Initial prediction (feature-level)
        prediction_features = self.initial_head(refined_features)

        # Iterative correction
        for _ in range(self.num_correction_rounds):
            upsampled_features = F.interpolate(
                refined_features, size=prediction_features.shape[-2:], mode='bilinear', align_corners=False
            )
            error_input = torch.cat([prediction_features, upsampled_features], dim=1)
            error_map = self.error_detector(error_input)
            correction_input = torch.cat([prediction_features, upsampled_features, error_map], dim=1)
            correction = self.corrector(correction_input)
            prediction_features = prediction_features + error_map * correction

        # Final segmentation logits
        logits = self.seg_head(F.interpolate(prediction_features, size=(H, W), mode='bilinear', align_corners=False))
        return logits  # (B, num_classes, H, W)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class EnhancedDeepLabV3Plus(nn.Module):
    """Enhanced DeepLabV3+ with improved ASPP and decoder."""
    
    def __init__(self, num_classes, encoder_weights="imagenet"):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            self._make_layer(64, 64, 3, stride=1),
            self._make_layer(64, 128, 4, stride=2),
            self._make_layer(128, 256, 6, stride=1, dilation=2),
            self._make_layer(256, 512, 3, stride=1, dilation=4),
        )
        
        # Enhanced ASPP
        self.aspp = EnhancedASPP(512, 256, [6, 12, 18, 24])
        
        # Low-level feature projection
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, dilation=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, dilation, dilation))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, dilation, dilation))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Normalization
        if x.max() <= 1.0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
            x = (x - mean) / std
        
        # Feature extraction
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 3:
                low_level_features = x
            features.append(x)
        
        high_level_features = features[-1]
        
        # ASPP
        aspp_features = self.aspp(high_level_features)
        
        # Decoder
        aspp_upsampled = F.interpolate(
            aspp_features, size=low_level_features.shape[-2:], mode='bilinear', align_corners=False
        )
        low_level_proj = self.low_level_proj(low_level_features)
        decoder_input = torch.cat([aspp_upsampled, low_level_proj], dim=1)
        output = self.decoder(decoder_input)
        
        return F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class EnhancedASPP(nn.Module):
    """Enhanced Atrous Spatial Pyramid Pooling."""
    
    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__()
        
        self.atrous_convs = nn.ModuleList()
        
        # 1x1 conv
        self.atrous_convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions
        for rate in atrous_rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        self.project = nn.Sequential(
            nn.Conv2d(len(atrous_rates) * out_channels + 2 * out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        res = []
        for conv in self.atrous_convs:
            res.append(conv(x))
        
        pool_result = self.global_avg_pool(x)
        pool_result = F.interpolate(pool_result, size=x.shape[-2:], mode='bilinear', align_corners=False)
        res.append(pool_result)
        
        res = torch.cat(res, dim=1)
        return self.project(res)


def get_transformer_model(model_arch, num_classes, encoder="segformer-b0", encoder_weights="imagenet"):
    """
    Factory function to get advanced segmentation models.
    
    Args:
        model_arch (str): Architecture name
        num_classes (int): Number of output classes
        encoder (str): Encoder variant
        encoder_weights (str): Pretrained weights
    
    Returns:
        nn.Module: Segmentation model
    """
    
    if model_arch == "SegFormer":
        return SegFormerModel(num_classes, encoder, encoder_weights)
    elif model_arch == "ViTSeg":
        if encoder == "vit-base":
            embed_dim, num_heads, num_layers = 768, 12, 12
        elif encoder == "vit-small":
            embed_dim, num_heads, num_layers = 384, 6, 12
        else:
            embed_dim, num_heads, num_layers = 384, 6, 12
        return SimpleViTSegmentation(
            num_classes=num_classes, image_size=512, patch_size=16,
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
            encoder_weights=encoder_weights
        )
    elif model_arch == "MaskDINO":
        return MaskDINOModel(num_classes, encoder_weights)
    elif model_arch == "SCTNet":
        return SCTNetModel(num_classes, encoder_weights)
    elif model_arch == "EnhancedDeepLabV3Plus":
        return EnhancedDeepLabV3Plus(num_classes, encoder_weights)
    elif model_arch == "HybridCNNTransformer":
        return HybridCNNTransformer(num_classes, encoder_weights)
    else:
        raise ValueError(f"Unknown architecture: {model_arch}")


# Alternative: Hybrid CNN-Transformer model
class HybridCNNTransformer(nn.Module):
    """
    Hybrid model combining CNN feature extraction with Transformer processing.
    """
    
    def __init__(self, num_classes, encoder_weights="imagenet"):
        super().__init__()
        
        # CNN backbone (using ResNet-like structure)
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Feature extraction layers
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Transformer for global context
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, 4, stride=2, padding=1)
        )
        
        self.num_classes = num_classes
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create a residual layer."""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Prepare for transformer
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Transformer processing
        x_trans = self.transformer(x_flat)
        
        # Reshape back to spatial
        x = x_trans.transpose(1, 2).reshape(B, C, H, W)
        
        # Decode to segmentation map
        x = self.decoder(x)
        
        return x
    
    def predict(self, x):
        """Prediction method compatible with test pipeline."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
