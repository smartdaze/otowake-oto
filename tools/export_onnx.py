#!/usr/bin/env python3
"""
Demucs → ONNX 変換スクリプト (NNコア分離方式)

使い方:
    pip install demucs torch onnx onnxruntime einops
    python tools/export_onnx.py --model htdemucs_6s

ルール:
    - このスクリプトは OtoWake-ONNX/tools/ 内に配置
    - music-separator / music-separator-web には一切変更を加えない

アーキテクチャ:
    HTDemucs の forward() から ONNX 非互換な操作を除外したラッパーを使用。
    - STFT / iSTFT / view_as_complex は JavaScript 側で処理
    - ONNX モデルは NN コア (encoder/transformer/decoder) のみ含む
    入力: mag [B, C*2, Fr, T_spec], mix [B, C, T]
    出力: freq_out [B, S, C*2, Fr, T_spec], time_out [B, S, C, T]
"""

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn as nn


class HTDemucsONNXCore(nn.Module):
    """
    HTDemucs の ONNX 互換コア部分。

    STFT (→ 複素テンソル) と iSTFT (← view_as_complex) を除外し、
    encoder / transformer / decoder のみを含む。

    入力:
        mag  [B, C*2, Fr, T_spec] — STFT → view_as_real → permute → reshape 済みの実数テンソル
        mix  [B, C, T]            — 時間ブランチ用の波形
    出力:
        freq_out  [B, S, C*2, Fr, T_spec] — 周波数ブランチ出力 (実数)
        time_out  [B, S, C, T]            — 時間ブランチ出力 (実数)
    """

    def __init__(self, model):
        super().__init__()
        # 元モデルの全サブモジュールを参照
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.tencoder = model.tencoder
        self.tdecoder = model.tdecoder
        self.crosstransformer = getattr(model, "crosstransformer", None)
        self.freq_emb = model.freq_emb
        self.freq_emb_scale = getattr(model, "freq_emb_scale", 0)
        self.bottom_channels = model.bottom_channels
        self.channel_upsampler = getattr(model, "channel_upsampler", None)
        self.channel_downsampler = getattr(model, "channel_downsampler", None)
        self.channel_upsampler_t = getattr(model, "channel_upsampler_t", None)
        self.channel_downsampler_t = getattr(model, "channel_downsampler_t", None)
        self.sources = model.sources
        self.depth = model.depth

    def forward(self, mag, mix):
        from einops import rearrange

        x = mag
        B, C_mag, Fq, T = x.shape
        length = mix.shape[-1]

        # ---- 正規化 (周波数ブランチ) ----
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        # ---- 正規化 (時間ブランチ) ----
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # ---- エンコーダ ----
        saved = []
        saved_t = []
        lengths = []
        lengths_t = []
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        # ---- トランスフォーマー ----
        if self.crosstransformer:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t -> b c (f t)")
                x = self.channel_upsampler(x)
                x = rearrange(x, "b c (f t) -> b c f t", f=f)
                xt = self.channel_upsampler_t(xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = rearrange(x, "b c f t -> b c (f t)")
                x = self.channel_downsampler(x)
                x = rearrange(x, "b c (f t) -> b c f t", f=f)
                xt = self.channel_downsampler_t(xt)

        # ---- デコーダ ----
        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        # ---- 出力整形・逆正規化 ----
        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]

        return x, xt


def compute_spec_params(model, seg_samples):
    """STFT パラメータと spec のサイズを計算する"""
    nfft = model.nfft
    hop_length = model.hop_length
    le = int(math.ceil(seg_samples / hop_length))
    pad = hop_length // 2 * 3
    num_freq_bins = nfft // 2  # :-1 trim 後
    spec_time_frames = le      # 2:2+le trim 後
    return {
        "nfft": nfft,
        "hop_length": hop_length,
        "num_freq_bins": num_freq_bins,
        "spec_time_frames": spec_time_frames,
        "spec_pad": pad,
        "spec_le": le,
    }


def export_model(model_name: str, output_path: str, opset: int = 17) -> None:
    """Demucs モデルを ONNX 形式にエクスポートする"""
    from demucs.pretrained import get_model

    print(f"[1/7] モデル '{model_name}' を読み込み中...")
    bag_or_model = get_model(model_name)

    if hasattr(bag_or_model, "models"):
        print("       BagOfModels を検出 → 内部モデル[0] を使用")
        model = bag_or_model.models[0]
    else:
        model = bag_or_model

    model.eval()
    model.cpu()

    sr = model.samplerate
    segment_raw = getattr(model, "segment", Fraction(39, 5))
    segment = float(segment_raw)
    sources = list(model.sources) if hasattr(model, "sources") else []
    if isinstance(segment_raw, Fraction):
        seg_samples = int(segment_raw * sr)
    else:
        seg_samples = round(segment * sr)

    audio_channels = model.audio_channels
    cac = model.cac

    print(f"       サンプルレート: {sr}Hz")
    print(f"       セグメント長: {segment}s ({seg_samples} samples)")
    print(f"       ソース: {sources}")
    print(f"       モデル型: {type(model).__name__}")
    print(f"       cac: {cac}, audio_channels: {audio_channels}")

    if not cac:
        print("エラー: cac=False のモデルは現在未対応です", file=sys.stderr)
        sys.exit(1)

    # ---- STFT パラメータ計算 ----
    spec_params = compute_spec_params(model, seg_samples)
    nfft = spec_params["nfft"]
    num_freq = spec_params["num_freq_bins"]
    spec_t = spec_params["spec_time_frames"]

    print(f"[2/7] STFT パラメータ: nfft={nfft}, hop={spec_params['hop_length']}")
    print(f"       周波数ビン: {num_freq}, 時間フレーム: {spec_t}")

    # ---- テスト推論 (元モデル) ----
    print(f"[3/7] 元モデルでテスト推論中...")
    dummy_mix = torch.randn(1, audio_channels, seg_samples)
    with torch.no_grad():
        ref_output = model(dummy_mix)
    print(f"       元モデル出力形状: {list(ref_output.shape)}")

    # ---- ラッパー作成 & テスト推論 ----
    print(f"[4/7] ONNX ラッパーを作成してテスト推論中...")
    wrapper = HTDemucsONNXCore(model)
    wrapper.eval()

    # STFT → magnitude をここで計算 (エクスポート用ダミー入力)
    with torch.no_grad():
        from demucs.spec import spectro
        from demucs.hdemucs import pad1d

        hl = spec_params["hop_length"]
        le = spec_params["spec_le"]
        pad_amount = spec_params["spec_pad"]

        x_padded = pad1d(
            dummy_mix,
            (pad_amount, pad_amount + le * hl - seg_samples),
            mode="reflect",
        )
        z_complex = spectro(x_padded, nfft, hl)
        z_complex = z_complex[..., :-1, :]
        z_complex = z_complex[..., 2 : 2 + le]

        # _magnitude 相当 (cac=True)
        B, C, Fr, T_s = z_complex.shape
        z_real = torch.view_as_real(z_complex)
        dummy_mag = z_real.permute(0, 1, 4, 2, 3).reshape(B, C * 2, Fr, T_s)

    print(f"       mag 形状: {list(dummy_mag.shape)}")
    print(f"       mix 形状: {list(dummy_mix.shape)}")

    with torch.no_grad():
        freq_out, time_out = wrapper(dummy_mag, dummy_mix)
    print(f"       freq_out 形状: {list(freq_out.shape)}")
    print(f"       time_out 形状: {list(time_out.shape)}")

    # ---- 精度検証: ラッパー出力 → _mask → _ispec で元モデルと一致するか ----
    print(f"[5/7] 精度検証中...")
    with torch.no_grad():
        zout = model._mask(z_complex, freq_out)
        x_freq = model._ispec(zout, seg_samples)
        reconstructed = x_freq + time_out
    diff = (ref_output - reconstructed).abs().max().item()
    print(f"       最大誤差: {diff:.2e} (< 1e-4 なら OK)")
    if diff > 1e-3:
        print(f"       警告: 誤差が大きいです!")

    # ---- ONNX エクスポート ----
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[6/7] ONNX にエクスポート中... (opset={opset})")
    torch.onnx.export(
        wrapper,
        (dummy_mag, dummy_mix),
        output_path,
        input_names=["mag", "mix"],
        output_names=["freq_out", "time_out"],
        opset_version=opset,
        do_constant_folding=True,
    )

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"       出力: {output_path}")
    print(f"       サイズ: {file_size_mb:.1f} MB")

    # ---- onnxruntime 検証 ----
    print(f"[7/7] onnxruntime で検証中...")
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(output_path)
        ort_inputs = {
            "mag": dummy_mag.numpy(),
            "mix": dummy_mix.numpy(),
        }
        ort_freq, ort_time = sess.run(None, ort_inputs)
        freq_diff = abs(ort_freq - freq_out.numpy()).max()
        time_diff = abs(ort_time - time_out.numpy()).max()
        print(f"       ONNX Runtime freq 誤差: {freq_diff:.2e}")
        print(f"       ONNX Runtime time 誤差: {time_diff:.2e}")
        print("       onnxruntime 検証成功!")
    except Exception as e:
        print(f"       onnxruntime 検証失敗: {e}")

    # ---- メタデータ JSON ----
    meta_path = Path(output_path).with_suffix(".json")
    meta = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "sources": sources,
        "samplerate": sr,
        "segment": segment,
        "segment_samples": seg_samples,
        "audio_channels": audio_channels,
        "cac": cac,
        "nfft": nfft,
        "hop_length": spec_params["hop_length"],
        "num_freq_bins": num_freq,
        "spec_time_frames": spec_t,
        "spec_pad": spec_params["spec_pad"],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"       メタデータ: {meta_path}")
    print()
    print("準備完了! フロントエンドを起動して音源分離をお試しください。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demucs モデルを ONNX 形式に変換するツール (NNコア分離方式)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="htdemucs_6s",
        choices=["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi"],
        help="変換するモデル名 (デフォルト: htdemucs_6s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力パス (デフォルト: frontend/public/models/<model_name>.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset バージョン (デフォルト: 17)",
    )

    args = parser.parse_args()

    if args.output is None:
        project_root = Path(__file__).parent.parent
        args.output = str(
            project_root / "frontend" / "public" / "models" / f"{args.model}.onnx"
        )

    try:
        export_model(args.model, args.output, args.opset)
    except Exception as e:
        print(f"\nエラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        print("\nヒント:", file=sys.stderr)
        print("  pip install demucs torch onnx onnxruntime einops", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
