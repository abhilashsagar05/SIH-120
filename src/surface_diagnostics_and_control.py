# -*- coding: utf-8 -*-
"""
================================================================================
  BAGHEWALA HEAVY OIL FIELD — Well-to-Surface Digital Twin
  Module : src/surface_diagnostics_and_control.py
  Classes: DynamometerCNN, VFDController
  Author : Digital Twin Engineering Team (SIH-DS026)
  Version: 1.0.0

  Purpose
  -------
  Executes the autonomous Surface Control loop for the Baghewala field.
  This module bridges the gap between digital simulation and physical actuation:
  
  1. Deep Learning Diagnostics (DynamometerCNN):
     Processes generated 2D downhole dynamometer cards (Position vs. Load) 
     as image tensors. Utilizes a deep residual network (ResNet architecture) 
     to classify the mechanical state of the Sucker Rod Pump (SRP).
     Classes: [Healthy Pump, Fluid Pound, Rod Floating].

  2. Autonomous VFD Control (VFDController):
     Consumes the CNN diagnostic probabilities and triggers physical actuation
     via Variable Frequency Drives (VFD). Specifically handles the severe 
     Baghewala heavy oil constraint by implementing Asymmetric Speed Profiling
     when Rod Floating is detected.

  Asymmetric Speed Profiling:
  ---------------------------
  As the reservoir cools and viscosity spikes to ~10,000+ cP, the downward 
  gravity of the rod string is matched by the upward viscous drag of the fluid,
  causing "Rod Floating" and catastrophic compressive buckling. 
  
  To solve this autonomously, the VFD is commanded to:
    - SLOW the downstroke (matching fluid resistance, keeping rods in tension).
    - ACCELERATE the upstroke (maximising volumetric lift and exploiting the 
      shear-thinning rheology of the crude).
================================================================================
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("PyTorch is required for Surface Diagnostics. Run: pip install torch")

import numpy as np


# ===========================================================================
# Module-level Logger
# ===========================================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler(stream=sys.stderr)
    try:
        _handler.stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    _handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ===========================================================================
# §1  Convolutional Neural Network (ResNet Architecture)
# ===========================================================================

class ResidualBlock(nn.Module):
    """
    Standard Residual Block for deep feature extraction.
    Ensures gradients flow through deep layers without vanishing, which is 
    critical for learning subtle geometric distortions in dynamometer cards.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DynamometerCNN(nn.Module):
    """
    Deep Residual CNN for classifying dynamometer card image tensors.
    Configured for 1-channel grayscale inputs (the rendered 2D cards).
    
    Output Classes:
        0 : Healthy Pump
        1 : Fluid Pound (Incomplete pump fillage)
        2 : Rod Floating (Extreme viscous drag / buckling)
    """
    CLASSES = ["Healthy Pump", "Fluid Pound", "Rod Floating"]

    def __init__(self):
        super().__init__()
        self.in_channels = 16
        
        # Initial Convolution
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Residual Layers
        self.layer1 = self._make_layer(16, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(32, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(64, num_blocks=2, stride=2)
        
        # Classifier Head
        self.fc = nn.Linear(64, len(self.CLASSES))

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expects x of shape (Batch, 1, H, W).
        Returns Softmax probabilities of shape (Batch, 3).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Global Average Pooling adapts to any input resolution (e.g., 64x64 or 224x224)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        
        logits = self.fc(out)
        return torch.softmax(logits, dim=1)


# ===========================================================================
# §2  Autonomous VFD Control Loop
# ===========================================================================

@dataclass
class VFDState:
    """Represents the current Variable Frequency Drive command state."""
    upstroke_spm: float   = 6.0    # Strokes Per Minute (upward lift)
    downstroke_spm: float = 6.0    # Strokes Per Minute (downward reset)
    motor_torque_pct: float = 60.0 # Motor load limit %


class VFDController:
    """
    Executes the autonomous control policy by evaluating the CNN diagnostic 
    flags and commanding the wellhead VFD.
    """
    def __init__(self):
        self.state = VFDState()
        self.base_spm = 6.0
        self.min_spm = 1.0
        self.max_spm = 8.0
        
        logger.info("VFD Controller Initialised. Nominal Symmetric SPM: %.1f", self.base_spm)

    def _apply_asymmetric_profile(self, severity: float):
        """
        Calculates the asymmetric kinematic profiling to prevent rod float.
        Severity [0 to 1] dictates the aggressiveness of the split.
        """
        # Slow downstroke to maintain tension against viscous drag
        self.state.downstroke_spm = max(
            self.base_spm * (1.0 - (0.6 * severity)), 
            self.min_spm
        )
        # Fast aggressive upstroke to exploit shear-thinning and lift volume
        self.state.upstroke_spm = min(
            self.base_spm * (1.0 + (0.3 * severity)), 
            self.max_spm
        )
        self.state.motor_torque_pct = min(60.0 + (30.0 * severity), 100.0)

    def _apply_pound_mitigation(self, severity: float):
        """
        Mitigates Fluid Pound by slowing overall pump speed to allow reservoir 
        inflow to catch up with pump displacement.
        """
        reduction = 1.0 - (0.4 * severity)
        self.state.upstroke_spm = max(self.base_spm * reduction, self.min_spm)
        self.state.downstroke_spm = max(self.base_spm * reduction, self.min_spm)
        self.state.motor_torque_pct = 60.0

    def execute_control_loop(self, diagnostic_probs: List[float], well_id: str = "BGW-07") -> VFDState:
        """
        Evaluates the probabilities from the CNN and sets the physical VFD command.
        
        Args:
            diagnostic_probs: [P_healthy, P_pound, P_float]
        """
        p_healthy, p_pound, p_float = diagnostic_probs
        
        logger.info(
            "[%s] CNN Diagnostics | Healthy: %2.0f%% | Pound: %2.0f%% | Float: %2.0f%%",
            well_id, p_healthy * 100, p_pound * 100, p_float * 100
        )

        if p_float > 0.65:
            # Critical rod floating detected. Intervene immediately.
            logger.warning("[%s] ALERT: Rod Floating detected. Initiating Asymmetric VFD Profile.", well_id)
            self._apply_asymmetric_profile(severity=p_float)
            
        elif p_pound > 0.65:
            # Incomplete fillage detected.
            logger.warning("[%s] ALERT: Fluid Pound detected. Slowing symmetric SPM.", well_id)
            self._apply_pound_mitigation(severity=p_pound)
            
        else:
            # Healthy operation. Return to nominal state gradually.
            logger.info("[%s] State: Healthy. Maintaining nominal symmetric operation.", well_id)
            self.state.upstroke_spm = self.base_spm
            self.state.downstroke_spm = self.base_spm
            self.state.motor_torque_pct = 60.0

        logger.info(
            "[%s] VFD CMD Applied | Upstroke: %.1f SPM | Downstroke: %.1f SPM | Torque Lim: %.1f%%",
            well_id, self.state.upstroke_spm, self.state.downstroke_spm, self.state.motor_torque_pct
        )
        
        return self.state


# ===========================================================================
# §3  Module Smoke Test
# ===========================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("  Surface Diagnostics & Autonomous VFD Control -- Smoke Test")
    print("  Baghewala Well-to-Surface Digital Twin")
    print("=" * 72)

    # 1. Initialize CNN & VFD Controller
    print("\n[1] Initialising Deep Learning Model and VFD Controller...")
    cnn_model = DynamometerCNN()
    cnn_model.eval()  # Set to evaluation mode
    vfd_controller = VFDController()
    
    # 2. Generate Dummy Tensor to simulate a rendered Dynamometer Card Image
    # Image shape: (Batch Size, Channels, Height, Width) -> 1 batch, Grayscale, 64x64 pixels
    print("\n[2] Generating Dummy Dynamometer Card Image Tensors...")
    dummy_images = [
        torch.randn(1, 1, 64, 64), # Simulates a healthy card
        torch.randn(1, 1, 64, 64), # Simulates a card with severe compression (Rod Floating)
    ]
    
    # Force the dummy network to spit out highly biased probabilities for the sake of the smoke test
    # (Since it is untrained, we manually override the output to demonstrate the control loop)
    simulated_outputs = [
        torch.tensor([[0.92, 0.05, 0.03]]), # High probability Healthy
        torch.tensor([[0.05, 0.05, 0.90]]), # High probability Rod Floating
    ]

    # 3. Execution Loop Simulation
    print("\n[3] Executing Autonomous Control Loop Simulation...")
    for i, img_tensor in enumerate(dummy_images):
        print(f"\n--- Timestep {i+1} ---")
        
        # Forward pass (in reality, we use the model's true output. Here we use the mocked output)
        with torch.no_grad():
            _ = cnn_model(img_tensor) # Demonstrate it processes the tensor without crashing
            probs = simulated_outputs[i].numpy().flatten().tolist()
            
        # Command VFD
        vfd_controller.execute_control_loop(diagnostic_probs=probs)
        time.sleep(1) # Simulate processing time

    print("\n[PASS] Surface Diagnostics & Control smoke test completed successfully.\n")
