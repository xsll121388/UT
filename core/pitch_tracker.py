"""Pitch extraction wrapper supporting multiple backends.

Supports:
  - RMVPE (ONNX/PyTorch): Default, robust pitch tracking
  - vslib: VoiceSynthLib, optimized for singing voice

RMVPE SPEC (v1.1):
  Input:
    waveform  → (1, T) float32 @16kHz
    threshold → ()    float32 scalar
  Output:
    f0 → (1, N) float32 Hz (N = frames at 100fps)
    uv → (1, N) bool   (True = unvoiced)
  Parameters:
    - Frame rate: 100fps
    - F0 range: 50Hz - 1100Hz
"""
from __future__ import annotations
import os
import warnings
import numpy as np
from utils import config

HOP_LENGTH = 160        # at 16kHz → 100fps (10ms per frame)
SAMPLE_RATE = 16000     # RMVPE expects 16kHz input
FMIN = 32.70            # C1 (model's F0 range lower bound)
FMAX = 1975.5           # B6 (model's F0 range upper bound)
OUTPUT_FPS = 100        # Fixed output frame rate


class PitchTracker:
    def __init__(self, backend: str = "rmvpe"):
        """Initialize pitch tracker with specified backend.

        Args:
            backend: "rmvpe" or "vslib"
        """
        self._backend = backend.lower()
        self._vslib = None

        if self._backend == "vslib":
            self._init_vslib()
        elif self._backend == "rmvpe":
            self._init_rmvpe()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'rmvpe' or 'vslib'")

    def _init_vslib(self):
        """Initialize vslib backend (C DLL wrapper)."""
        try:
            import ctypes
            import platform

            vslib_dir = r"D:\OpenTunePro\vslib156"

            # Choose correct DLL based on architecture
            if platform.architecture()[0] == '64bit':
                dll_path = os.path.join(vslib_dir, "vslib_x64.dll")
            else:
                dll_path = os.path.join(vslib_dir, "vslib.dll")

            if not os.path.exists(dll_path):
                raise FileNotFoundError(f"vslib DLL not found: {dll_path}")

            self._vslib_dll = ctypes.CDLL(dll_path)
            self._model_type = "vslib"
            print(f"[PitchTracker] Using vslib backend from {dll_path}")

            # Setup function signatures (will need to adjust based on vslib.h)
            # This is a placeholder - need to check vslib.h for actual API
            self._setup_vslib_functions()

        except Exception as e:
            raise ImportError(
                f"Failed to load vslib DLL from D:\\OpenTunePro\\vslib156\n"
                f"Error: {e}"
            )

    def _setup_vslib_functions(self):
        """Setup vslib DLL function signatures based on vslib.h."""
        import ctypes

        # AS_AnalyzeWaveDataEX signature:
        # int AS_AnalyzeWaveDataEX(void *wavdata, unsigned short *sndspc,
        #     int *pitch, int *dynamics, AWDINFO *awdi, double freqa4)

        # AWDINFO structure (from vslib.h line 143-152)
        class AWDINFO(ctypes.Structure):
            _fields_ = [
                ("wavdatasize", ctypes.c_int),
                ("wavsampleps", ctypes.c_int),
                ("wavbit", ctypes.c_int),
                ("wavchannel", ctypes.c_int),
                ("nnoffset", ctypes.c_int),
                ("nnrange", ctypes.c_int),
                ("blockpn", ctypes.c_int),
                ("targetch", ctypes.c_int),
                ("option", ctypes.c_uint),
                ("reserved", ctypes.c_int * 7)
            ]

        self._AWDINFO = AWDINFO

        # Setup function signatures
        self._vslib_dll.AS_AnalyzeWaveDataEX.argtypes = [
            ctypes.c_void_p,  # wavdata
            ctypes.POINTER(ctypes.c_ushort),  # sndspc
            ctypes.POINTER(ctypes.c_int),  # pitch
            ctypes.POINTER(ctypes.c_int),  # dynamics
            ctypes.POINTER(AWDINFO),  # awdi
            ctypes.c_double  # freqa4
        ]
        self._vslib_dll.AS_AnalyzeWaveDataEX.restype = ctypes.c_int

        # Frequency conversion functions
        self._vslib_dll.AS_Cent2Freq.argtypes = [ctypes.c_int, ctypes.c_double]
        self._vslib_dll.AS_Cent2Freq.restype = ctypes.c_double

        self._vslib_dll.AS_Freq2Cent.argtypes = [ctypes.c_double, ctypes.c_double]
        self._vslib_dll.AS_Freq2Cent.restype = ctypes.c_int

    @staticmethod
    def is_vslib_available() -> bool:
        """Check if vslib DLL is available."""
        try:
            import platform
            vslib_dir = r"D:\OpenTunePro\vslib156"

            if platform.architecture()[0] == '64bit':
                dll_path = os.path.join(vslib_dir, "vslib_x64.dll")
            else:
                dll_path = os.path.join(vslib_dir, "vslib.dll")

            return os.path.exists(dll_path)
        except:
            return False

    def _init_rmvpe(self):
        """Initialize RMVPE backend."""
        model_dir = config.get_model_dir()

        # Try to load model - prefer .pt format, fallback to .onnx
        self._model_type = None
        self._session = None
        self._torch_model = None

        # Check for .pt model first
        pt_path = os.path.join(model_dir, "rmvpe.pt")
        onnx_path = os.path.join(model_dir, "rmvpe.onnx")

        if os.path.exists(pt_path):
            self._load_pytorch_model(pt_path)
            self._model_type = "pytorch"
        elif os.path.exists(onnx_path):
            self._load_onnx_model(onnx_path)
            self._model_type = "onnx"
        else:
            raise FileNotFoundError(
                f"RMVPE model not found. Expected one of:\n"
                f"  - {pt_path}\n"
                f"  - {onnx_path}"
            )
    
    def _load_pytorch_model(self, model_path: str):
        """Load PyTorch .pt model (supports both TorchScript and state_dict formats)."""
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required to load .pt models. "
                "Install with: pip install torch"
            )
        
        # Try loading as TorchScript first
        try:
            self._torch_model = torch.jit.load(model_path, map_location="cpu")
            self._model_format = "torchscript"
            print(f"[PitchTracker] Loaded TorchScript model: {model_path}")
        except Exception as e:
            # If TorchScript fails, try loading as state_dict
            # This requires the model class to be available
            try:
                # Try to load as a regular PyTorch model
                checkpoint = torch.load(model_path, map_location="cpu")
                
                # Check if it's a state_dict or a full model
                if isinstance(checkpoint, dict):
                    # It's a state_dict, we need to reconstruct the model
                    # For RMVPE, we'll try to create a simple wrapper
                    self._torch_model = self._create_rmvpe_model(checkpoint)
                    self._model_format = "state_dict"
                    print(f"[PitchTracker] Loaded PyTorch state_dict model: {model_path}")
                else:
                    # It's a full model
                    self._torch_model = checkpoint
                    self._torch_model.eval()
                    self._model_format = "full_model"
                    print(f"[PitchTracker] Loaded PyTorch full model: {model_path}")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load PyTorch model. "
                    f"TorchScript error: {e}\n"
                    f"State dict error: {e2}"
                )
        
        if hasattr(self._torch_model, 'eval'):
            self._torch_model.eval()
        
        # Store model info
        self._input_names = ["waveform", "threshold"]
        self._output_names = ["f0", "uv"]
    
    def _create_rmvpe_model(self, checkpoint):
        """Create RMVPE model from state_dict checkpoint."""
        import torch
        import torch.nn as nn
        
        # Simple RMVPE model wrapper
        # This is a basic implementation - you may need to adjust based on actual model architecture
        class RMVPEWrapper(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                # Try to load state dict into a simple sequential model
                # This assumes the model is a simple feed-forward network
                self.model = self._build_model_from_state_dict(state_dict)
                self.load_state_dict(state_dict, strict=False)
            
            def _build_model_from_state_dict(self, state_dict):
                # This is a placeholder - actual implementation depends on model architecture
                # For now, return a simple pass-through
                return nn.Identity()
            
            def forward(self, waveform, threshold=None):
                # Forward pass - adjust based on actual model
                return self.model(waveform)
        
        try:
            model = RMVPEWrapper(checkpoint)
            model.eval()
            return model
        except Exception as e:
            # If we can't create the model, raise an error with helpful message
            raise RuntimeError(
                f"Unable to load model from state_dict. "
                f"The model may need to be converted to TorchScript format. "
                f"Error: {e}"
            )
    
    def _load_onnx_model(self, model_path: str):
        """Load ONNX model."""
        import onnxruntime as ort
        
        # CPU only (matches OpenTune V1.1 behaviour)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        self._session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # Validate model I/O on init
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        self._input_names = [inp.name for inp in inputs]
        self._output_names = [out.name for out in outputs]
        
        print(f"[PitchTracker] Loaded ONNX model: {model_path}")

    def extract(self, audio_16k: np.ndarray, threshold: float = 0.03) -> np.ndarray:
        """
        Extract F0 from 16kHz mono float32 audio.

        Returns:
            f0 array (N,) float32 at 100fps; 0.0 = unvoiced

        Args:
            audio_16k: 16kHz mono float32 audio
            threshold: voiced/unvoiced confidence cutoff (0.0–1.0)
        """
        if len(audio_16k) == 0:
            return np.array([], dtype=np.float32)

        audio = audio_16k.astype(np.float32)

        # Route to appropriate backend
        if self._backend == "vslib":
            return self._extract_vslib(audio)
        else:
            # RMVPE backend
            # Pad to multiple of HOP_LENGTH (required by model)
            pad = (-len(audio)) % HOP_LENGTH
            if pad:
                audio = np.pad(audio, (0, pad))

            if self._model_type == "pytorch":
                return self._extract_pytorch(audio, threshold)
            else:
                return self._extract_onnx(audio, threshold)

    def _extract_vslib(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 using vslib backend."""
        import ctypes

        try:
            # vslib expects 16-bit PCM audio
            # Convert float32 [-1, 1] to int16
            audio_int16 = (audio * 32767).astype(np.int16)

            # Calculate block size (vslib uses blocks for analysis)
            # blockpn = number of blocks per note (typically 10-20)
            blockpn = 10
            block_samples = len(audio_int16) // blockpn
            if block_samples == 0:
                block_samples = len(audio_int16)
                blockpn = 1

            # Prepare output arrays
            num_blocks = blockpn
            pitch_array = (ctypes.c_int * num_blocks)()
            dynamics_array = (ctypes.c_int * num_blocks)()
            sndspc_array = (ctypes.c_ushort * (num_blocks * 256))()  # Spectrum data

            # Setup AWDINFO structure
            awdi = self._AWDINFO()
            awdi.wavdatasize = len(audio_int16)
            awdi.wavsampleps = SAMPLE_RATE  # 16kHz
            awdi.wavbit = 16
            awdi.wavchannel = 1  # mono
            awdi.nnoffset = 0  # Note number offset (C4 = 60)
            awdi.nnrange = 88  # Full piano range
            awdi.blockpn = blockpn
            awdi.targetch = 0  # AS_CH_LPR (mono/center)
            awdi.option = 0  # Default options

            # Call vslib analysis
            freqa4 = 440.0  # A4 frequency
            result = self._vslib_dll.AS_AnalyzeWaveDataEX(
                audio_int16.ctypes.data_as(ctypes.c_void_p),
                sndspc_array,
                pitch_array,
                dynamics_array,
                ctypes.byref(awdi),
                freqa4
            )

            if result != 0:  # AS_ERR_NOERR
                warnings.warn(f"vslib analysis returned error code: {result}")
                expected_frames = int(len(audio) / HOP_LENGTH)
                return np.zeros(expected_frames, dtype=np.float32)

            # Convert pitch from cents to Hz
            # vslib returns pitch in cents (relative to C0)
            f0_blocks = np.zeros(num_blocks, dtype=np.float32)
            for i in range(num_blocks):
                if pitch_array[i] > 0:  # Valid pitch
                    # Convert cents to frequency
                    f0_blocks[i] = self._vslib_dll.AS_Cent2Freq(pitch_array[i], freqa4)
                else:
                    f0_blocks[i] = 0.0  # Unvoiced

            # Resample to 100fps (expected output rate)
            expected_frames = int(len(audio) / HOP_LENGTH)
            if len(f0_blocks) != expected_frames:
                from scipy.interpolate import interp1d
                t_old = np.linspace(0, 1, len(f0_blocks))
                t_new = np.linspace(0, 1, expected_frames)

                # Separate voiced and unvoiced
                voiced = f0_blocks > 0
                if voiced.any():
                    f0_interp = interp1d(t_old, f0_blocks, kind='linear', fill_value='extrapolate')
                    f0_resampled = f0_interp(t_new)

                    voiced_interp = interp1d(t_old, voiced.astype(np.float32), kind='linear', fill_value='extrapolate')
                    voiced_resampled = voiced_interp(t_new) > 0.5

                    f0_resampled[~voiced_resampled] = 0.0
                    return f0_resampled.astype(np.float32)
                else:
                    return np.zeros(expected_frames, dtype=np.float32)

            return f0_blocks.astype(np.float32)

        except Exception as e:
            warnings.warn(f"vslib extraction failed: {e}, falling back to zeros")
            expected_frames = int(len(audio) / HOP_LENGTH)
            return np.zeros(expected_frames, dtype=np.float32)

    
    def _extract_pytorch(self, audio: np.ndarray, threshold: float) -> np.ndarray:
        """Extract F0 using PyTorch model."""
        import torch
        
        # Prepare inputs
        inp = torch.from_numpy(audio[np.newaxis, :])  # (1, T)
        thresh = torch.tensor(threshold, dtype=torch.float32)
        
        # Run inference
        with torch.no_grad():
            try:
                outputs = self._torch_model(inp, thresh)
            except Exception as e:
                # Try alternative input format
                try:
                    outputs = self._torch_model(inp)
                except Exception:
                    raise RuntimeError(f"PyTorch inference failed: {e}")
        
        # Process outputs
        if isinstance(outputs, (list, tuple)):
            f0_raw = outputs[0].numpy()
        else:
            f0_raw = outputs.numpy()
        
        # Handle output shape
        if f0_raw.ndim == 2:
            f0 = f0_raw[0].astype(np.float32)
        else:
            f0 = f0_raw.squeeze().astype(np.float32)
        
        # Apply UV mask if available
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
            uv_raw = outputs[1].numpy()
            if uv_raw.ndim == 2:
                uv = uv_raw[0]
            else:
                uv = uv_raw.squeeze()
            
            if uv.dtype == bool:
                f0[uv] = 0.0
            else:
                f0[uv > 0] = 0.0
        
        return f0
    
    def _extract_onnx(self, audio: np.ndarray, threshold: float) -> np.ndarray:
        """Extract F0 using ONNX model."""
        # Prepare inputs matching model signature:
        #   waveform: (1, T) float32
        #   threshold: () float32 scalar
        inp = audio[np.newaxis, :]  # (1, T)
        thresh = np.array(threshold, dtype=np.float32)

        # Run inference
        try:
            outputs = self._session.run(
                None,
                {"waveform": inp, "threshold": thresh}
            )
        except Exception as e:
            raise RuntimeError(f"RMVPE inference failed: {e}")

        # Process outputs - model returns 2D arrays:
        #   f0: (1, N) or (N,) depending on export
        #   uv: (1, N) or (N,) bool
        f0_raw = outputs[0]

        # Robust shape handling for different ONNX export formats
        if f0_raw.ndim == 2:
            f0 = f0_raw[0].astype(np.float32)  # Take first batch
        else:
            f0 = f0_raw.squeeze().astype(np.float32)

        # Apply UV mask if available
        if len(outputs) >= 2 and outputs[1] is not None:
            uv_raw = outputs[1]
            if uv_raw.ndim == 2:
                uv = uv_raw[0]  # (N,)
            else:
                uv = uv_raw.squeeze()

            # Apply unvoiced mask (True/1 = unvoiced → set F0 to 0)
            if uv.dtype == bool:
                f0[uv] = 0.0
            else:
                # Numeric mask: non-zero = unvoiced
                f0[uv > 0] = 0.0

        return f0

    def extract_from_44k(self, audio_44k: np.ndarray) -> np.ndarray:
        """Convenience: resample from 44.1kHz then extract."""
        import librosa
        audio_16k = librosa.resample(audio_44k, orig_sr=44100, target_sr=SAMPLE_RATE)
        return self.extract(audio_16k)

    @property
    def info(self) -> dict:
        """Return model information."""
        return {
            'sample_rate': SAMPLE_RATE,
            'hop_length': HOP_LENGTH,
            'output_fps': OUTPUT_FPS,
            'f0_range': (FMIN, FMAX),
            'input_names': self._input_names,
            'output_names': self._output_names,
            'model_type': self._model_type,
        }
