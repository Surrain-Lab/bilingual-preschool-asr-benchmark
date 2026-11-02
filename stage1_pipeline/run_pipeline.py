"""
run_pipeline.py
===============
End-to-end pipeline orchestrator.

Give it audio + annotated text → get Excel comparison reports.

Usage:
    python run_pipeline.py                          # uses defaults: audio/ and annotated-text/
    python run_pipeline.py --audio /path/to/audio --annotated /path/to/annotated-text

Optional:
    --output      outputs/           Override output root (default: outputs/)
    --pipelines   vtc1 vtc2 whisperx Run specific pipelines (default: all)
    --steps       0 1 2 3 4 5        Run specific steps only (default: all)
    --vtc1-model  /path/to/voice_type_classifier   VTC1 model dir (default: VTC1_Pipeline/voice_type_classifier)
    --vtc2-model  /path/to/New_VTC/VTC             VTC2 model dir
    --language    Spanish            Transcript language (default: Spanish)
    --device      cpu                Device for VTC2 inference: cpu/cuda/gpu (default: cpu)
    --dry-run                        Print commands without running

Steps:
    0      Human eval reference generation
    pre-1  Auto-convert all WAVs to 16kHz mono (runs automatically before step 1)
    1      VTC model inference + transcription (all 3 pipelines)
    2      Per-clip WER evaluation
    3      Per-recording WER aggregation
    4      Language distribution analysis (langdetect + word count)
    5      Excel report generation

VTC1 prerequisites:
    conda activate pyannote   (environment with pyannote-audio and sox)

VTC2 prerequisites:
    uv installed  (https://github.com/astral-sh/uv)
    Run from the VTC2 model directory
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Optional: prepend the conda env's lib/ dir to LD_LIBRARY_PATH if ctranslate2
# cannot find cuDNN. Left off by default, because if the env's libstdc++.so.6 is
# older than the system's it breaks ffmpeg/ffprobe (WhisperX chunking) and
# torchcodec in the VTC2 venv.
# _conda_lib = str(Path(sys.executable).parent.parent / "lib")
# os.environ["LD_LIBRARY_PATH"] = (
#     _conda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
# ).rstrip(":")

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
PROJ     = Path(__file__).parent
HUM      = PROJ / "Human_Evaluation"
VTC1     = PROJ / "VTC1_Pipeline"
VTC2     = PROJ / "VTC2_Pipeline"
WHISPERX = PROJ / "Only_Whisperx"


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def run(cmd, step_name, dry=False, cwd=None):
    """Run a subprocess command with clear logging."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n  $ {cmd_str}")
    if cwd:
        print(f"    (cwd: {cwd})")
    if dry:
        return True
    result = subprocess.run(cmd, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"\n  ERROR: FAILED: {step_name}")
        return False
    return True


def step_header(n, name):
    print(f"\n{'='*70}")
    print(f"  STEP {n}: {name}")
    print(f"{'='*70}")


def check(path, label):
    """Check a required path exists."""
    if not Path(path).exists():
        print(f"  WARNING: {label} not found: {path}")
        return False
    return True


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# STEP 0 — Human Eval Reference Generation
# ─────────────────────────────────────────────────────────────────

def step0_human_eval(cfg, dry):
    step_header(0, "Human Eval — Reference Generation")
    ok = True

    # 0a: Convert CHAT transcripts → clean reference text
    print("\n  [0a] Converting human transcripts to clean reference text...")
    ok &= run([
        sys.executable,
        str(HUM / "transcript_converter2.py"),
        "-i", cfg.annotated,
        "-o", cfg.o.human.benchmark,
    ], "transcript_converter2", dry)

    # 0b: Combine per-clip → per-recording reference files
    print("\n  [0b] Combining clips into per-recording reference files...")
    ok &= run([
        sys.executable,
        str(HUM / "combine_clips2.py"),
        "-i", cfg.o.human.benchmark,
        "-o", cfg.o.human.combined,
    ], "combine_clips2", dry)

    # 0c: Word count on human transcripts
    print("\n  [0c] Counting English/Spanish words in human transcripts...")
    ok &= run([
        sys.executable,
        str(HUM / "count_eng_esp_final.py"),
        "-i", cfg.annotated,
        "-o", cfg.o.human.word_count,
    ], "count_eng_esp_final", dry)

    return ok


# ─────────────────────────────────────────────────────────────────
# PRE-STEP 1 — Audio Format Normalisation (16kHz mono)
# ─────────────────────────────────────────────────────────────────

def ensure_16k_mono(audio_dir, dry=False):
    """Convert all WAV files under audio_dir to 16kHz mono in-place if needed.

    Uses ffprobe to check each file first — files that are already 16kHz mono
    are left untouched. Conversion writes to a temp file then atomically
    replaces the original so a failed run never corrupts source files.
    """
    audio_path = Path(audio_dir)
    wav_files  = sorted(audio_path.rglob("*.wav"))

    if not wav_files:
        print("  WARNING: No WAV files found in audio directory.")
        return True

    print(f"\n  Checking {len(wav_files)} WAV file(s) for 16kHz mono format...")

    converted = skipped = failed = 0

    for wav in wav_files:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels",
                "-of", "csv=p=0",
                str(wav),
            ],
            capture_output=True, text=True,
        )
        if probe.returncode != 0:
            print(f"    WARNING: Could not probe {wav.name} — skipping")
            skipped += 1
            continue

        parts = probe.stdout.strip().split(",")
        if len(parts) != 2:
            print(f"    WARNING: Unexpected ffprobe output for {wav.name} — skipping")
            skipped += 1
            continue

        sample_rate, channels = parts[0].strip(), parts[1].strip()

        if sample_rate == "16000" and channels == "1":
            skipped += 1
            continue

        print(f"    Converting {wav.name}  ({sample_rate} Hz, {channels}ch  →  16000 Hz, 1ch)")
        converted += 1
        if dry:
            continue

        tmp = wav.with_suffix(".tmp.wav")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav), "-ar", "16000", "-ac", "1", str(tmp)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"    ERROR: Conversion failed for {wav.name}")
            tmp.unlink(missing_ok=True)
            failed += 1
            continue

        tmp.replace(wav)

    status = f"  Audio check done: {converted} converted, {skipped} already OK"
    if failed:
        status += f", {failed} FAILED"
    print(status)
    return failed == 0


# ─────────────────────────────────────────────────────────────────
# STEP 1 — Transcription
# ─────────────────────────────────────────────────────────────────

def step1_whisperx(cfg, dry):
    step_header(1, "WhisperX-Only — Transcription")
    ok = True

    # 1C-1: Chunk + transcribe
    print("\n  [1C-1] Chunking and transcribing with WhisperX large-v3...")
    ok &= run([
        sys.executable,
        str(WHISPERX / "whisperx_detect_lang.py"),
        "--audio",   cfg.audio,
        "--chunks",  cfg.o.whisperx.chunks,
        "--output",  cfg.o.whisperx.raw_transcripts,
    ], "whisperx_detect_lang", dry)

    # 1C-2: Merge chunks
    print("\n  [1C-2] Merging chunk transcripts...")
    ok &= run([
        sys.executable,
        str(WHISPERX / "merge_whisperx_generated_chunk_step2.py"),
        "-i", cfg.o.whisperx.raw_transcripts,
        "-o", cfg.o.whisperx.merged,
    ], "merge_chunks", dry)

    # 1C-3: Assign speaker labels using VTC1 RTTM timestamps
    # 1C-3: Assign speaker labels using VTC1 RTTM txt (shared with VTC1 pipeline)
    # VTC1 must have been run first (step 1 with vtc1 in --pipelines)
    rttm_txt = cfg.o.vtc1.rttm_txt
    if Path(rttm_txt).exists() and any(Path(rttm_txt).rglob("*.txt")):
        print("\n  [1C-3] Assigning speaker labels using VTC1 RTTM timestamps...")
        ok &= run([
            sys.executable,
            str(WHISPERX / "transcript_based_speaker_tagger_step3.py"),
            "--speaker-dir", rttm_txt,
            "--source-dir",  cfg.o.whisperx.merged,
            "--output-dir",  cfg.o.whisperx.tagged,
        ], "speaker_tagger", dry)
    else:
        print("  WARNING: VTC1 RTTM txt not yet generated — speaker tagging skipped")
        print("      Run VTC1 pipeline first: --pipelines vtc1 --steps 1")
        print(f"      Expected at: {rttm_txt}")

    return ok


def step1_vtc1(cfg, dry):
    step_header(1, "VTC1 + WhisperX — Model Inference + Transcription")

    vtc1_model = Path(cfg.vtc1_model)
    if not vtc1_model.exists():
        print(f"  WARNING: VTC1 model directory not found: {vtc1_model}")
        print(f"      Clone from: https://github.com/MarvinLvn/voice-type-classifier")
        return False

    apply_sh = vtc1_model / "apply.sh"
    if not apply_sh.exists():
        print(f"  ERROR: apply.sh not found in {vtc1_model}")
        return False

    # 1A-0: Run VTC1 model on each recording folder
    # VTC1 apply.sh takes a folder of WAVs and produces:
    #   voice_type_classifier/output_voice_type_classifier/{recording_name}/all.rttm
    audio_path = Path(cfg.audio)
    recording_folders = sorted([d for d in audio_path.iterdir() if d.is_dir()])

    if not recording_folders:
        # If audio dir has no subfolders, treat it as a single recording
        recording_folders = [audio_path]

    vtc1_rttm_base = vtc1_model / "output_voice_type_classifier"
    ok = True

    print(f"\n  [1A-0] Running VTC1 model on {len(recording_folders)} recording folder(s)...")
    print(f"         Working dir: {vtc1_model}")
    for rec_folder in recording_folders:
        print(f"\n  Processing: {rec_folder.name}")
        vtc1_device = "gpu" if cfg.device in ("cuda", "gpu") else "cpu"
        ok &= run(
            ["bash", str(apply_sh), str(rec_folder), f"--device={vtc1_device}"],
            f"VTC1 apply.sh — {rec_folder.name}",
            dry,
            cwd=str(vtc1_model),
        )

    # 1A-1: RTTM → txt
    print("\n  [1A-1] Converting VTC1 RTTM to timestamped text...")
    ok &= run([
        sys.executable,
        str(VTC1 / "rttm_txt_unmerged.py"),
        "-i", str(vtc1_rttm_base),
        "-o", cfg.o.vtc1.rttm_txt,
    ], "rttm_txt_unmerged", dry)

    # 1A-2: Transcribe segments with WhisperX
    print("\n  [1A-2] Transcribing VTC1 segments with WhisperX large-v3...")
    ok &= run([
        sys.executable,
        str(VTC1 / "whisperx_vtc_transcribe_unmerged.py"),
        "--audio",  cfg.audio,
        "--rttm",   cfg.o.vtc1.rttm_txt,
        "--output", cfg.o.vtc1.tagged,
    ], "whisperx_vtc_transcribe_unmerged", dry)

    # 1A-3: Split by speaker
    print("\n  [1A-3] Splitting VTC1 transcripts by speaker...")
    ok &= run([
        sys.executable,
        str(VTC1 / "split_transcript_step2_unmerged.py"),
        "-i", cfg.o.vtc1.tagged,
        "-o", cfg.o.vtc1.speaker_separated,
    ], "split_transcript_step2_unmerged", dry)

    return ok


def step1_vtc2(cfg, dry):
    step_header(1, "VTC2 + WhisperX — Model Inference + Transcription")

    # Coordination guard: when VTC2 is being produced by a separate parallel run
    # (e.g. on another GPU), that run drops a marker so this orchestrator does NOT
    # also run VTC2 (which would collide on outputs/vtc2). The parallel run itself
    # sets FORCE_VTC2=1 to bypass this guard. Marker is removed when parallel done.
    if (PROJ / ".vtc2_handled_externally").exists() and not os.environ.get("FORCE_VTC2"):
        print("    VTC2 step 1 skipped — handled by a parallel run "
              "(.vtc2_handled_externally present; set FORCE_VTC2=1 to override).")
        return True

    vtc2_model = Path(cfg.vtc2_model)
    if not vtc2_model.exists():
        print(f"  WARNING: VTC2 model directory not found: {vtc2_model}")
        print(f"      Clone from: https://github.com/google-research/sound-separation")
        print(f"      Or set --vtc2-model to your VTC2 directory")
        return False

    infer_script = vtc2_model / "scripts" / "infer.py"
    config_file  = vtc2_model / "VTC-2.0" / "model" / "config.yml"
    checkpoint   = vtc2_model / "VTC-2.0" / "model" / "best.ckpt"

    for f, label in [(infer_script, "infer.py"), (config_file, "config.yml"), (checkpoint, "best.ckpt")]:
        if not f.exists():
            print(f"  ERROR: {label} not found: {f}")
            return False

    ok = True
    vtc2_rttm_output = cfg.o.vtc2.rttm_raw

    # 1B-0: Run VTC2 model inference
    print(f"\n  [1B-0] Running VTC2 model inference...")
    print(f"         Audio: {cfg.audio}")
    print(f"         Output: {vtc2_rttm_output}")
    ok &= run(
        [
            "uv", "run", str(infer_script),
            "--wavs",       cfg.audio,
            "--output",     vtc2_rttm_output,
            "--config",     str(config_file),
            "--checkpoint", str(checkpoint),
            "--device",     cfg.device,
            "--keep_raw",                      # keeps raw_rttm/ subfolder
            "--recursive_search",              # handles nested recording folders
        ],
        "VTC2 infer.py",
        dry,
        cwd=str(vtc2_model),
    )

    # VTC2 outputs to {vtc2_rttm_output}/raw_rttm/
    vtc2_raw_rttm = str(Path(vtc2_rttm_output) / "raw_rttm")

    # 1B-1: RTTM → txt
    print("\n  [1B-1] Converting VTC2 RTTM to timestamped text...")
    ok &= run([
        sys.executable,
        str(VTC2 / "rttm_txt_updated.py"),
        "-i", vtc2_raw_rttm,
        "-o", cfg.o.vtc2.rttm_txt,
    ], "rttm_txt_updated", dry)

    # 1B-2: Transcribe segments with WhisperX
    print("\n  [1B-2] Transcribing VTC2 segments with WhisperX large-v3...")
    ok &= run([
        sys.executable,
        str(VTC2 / "whisperx_vtc2_transcrib.py"),
        "--audio",  cfg.audio,
        "--rttm",   cfg.o.vtc2.rttm_txt,
        "--output", cfg.o.vtc2.tagged,
    ], "whisperx_vtc2_transcrib", dry)

    # 1B-3: Split by speaker
    print("\n  [1B-3] Splitting VTC2 transcripts by speaker...")
    ok &= run([
        sys.executable,
        str(VTC2 / "labels_seperator.py"),
        "-i", cfg.o.vtc2.tagged,
        "-o", cfg.o.vtc2.speaker_separated,
    ], "labels_seperator", dry)

    return ok


# ─────────────────────────────────────────────────────────────────
# STEP 2 — Per-Clip WER Evaluation
# ─────────────────────────────────────────────────────────────────

def step2_evaluate(cfg, pipelines, dry):
    step_header(2, "Per-Clip WER Evaluation")
    ok = True

    eval_configs = {
        "vtc1": (
            str(VTC1 / "evaluation_unmerged_not_applicable.py"),
            cfg.o.vtc1.speaker_separated,
            cfg.o.vtc1.eval_report,
        ),
        "vtc2": (
            str(VTC2 / "evaluation.py"),
            cfg.o.vtc2.speaker_separated,
            cfg.o.vtc2.eval_report,
        ),
        "whisperx": (
            str(WHISPERX / "evaluation_step4.py"),
            cfg.o.whisperx.tagged,
            cfg.o.whisperx.eval_report,
        ),
    }

    for pipe_name, (script, hyp_dir, out_dir) in eval_configs.items():
        if pipe_name not in pipelines:
            continue
        if not check(hyp_dir, f"{pipe_name} speaker-separated transcripts"):
            print(f"  Skipping {pipe_name} evaluation — transcription output not found")
            continue

        print(f"\n  Evaluating {pipe_name}...")
        mkdir(out_dir)
        ok &= run([
            sys.executable, script,
            "--ground-truth-dir", cfg.o.human.benchmark,
            "--whisperx-dir",     hyp_dir,
            "--output-dir",       out_dir,
            "--language",         cfg.language,
        ], f"evaluation_{pipe_name}", dry)

    return ok


# ─────────────────────────────────────────────────────────────────
# STEP 3 — Per-Recording WER Aggregation (sum counts, not average)
# ─────────────────────────────────────────────────────────────────

def step3_aggregate(cfg, pipelines, dry):
    step_header(3, "Per-Recording WER Aggregation")
    ok = True

    agg_configs = {
        "vtc1":    (str(VTC1 / "combine_calculation_unmerged_overall_final_eval.py"), cfg.o.vtc1.eval_report),
        "vtc2":    (str(VTC2 / "combine_calculation_overall_final_eval.py"),           cfg.o.vtc2.eval_report),
        "whisperx":(str(WHISPERX / "combine_calculation_overall_final_eval.py"),       cfg.o.whisperx.eval_report),
    }

    for pipe_name, (script, folder) in agg_configs.items():
        if pipe_name not in pipelines:
            continue
        if not check(folder, f"{pipe_name} eval_report folder"):
            continue

        print(f"\n  Aggregating {pipe_name}...")
        ok &= run([
            sys.executable, script,
            "--folder", folder,
        ], f"combine_calculation_{pipe_name}", dry)

    return ok


# ─────────────────────────────────────────────────────────────────
# STEP 4 — Language Distribution Analysis
# ─────────────────────────────────────────────────────────────────

def step4_langdetect(cfg, pipelines, dry):
    step_header(4, "Language Distribution Analysis")
    ok = True

    langdetect_configs = {
        "vtc1": (
            str(VTC1 / "langdetect_fallback" / "langdetect_tags_unmerged.py"),
            str(VTC1 / "langdetect_fallback" / "count_words_unmerged.py"),
            cfg.o.vtc1.speaker_separated,
            cfg.o.vtc1.langdetect,
            cfg.o.vtc1.word_count,
        ),
        "vtc2": (
            str(VTC2 / "Langdetect_fallback" / "langdetect_tags.py"),
            str(VTC2 / "Langdetect_fallback" / "count_words.py"),
            cfg.o.vtc2.speaker_separated,
            cfg.o.vtc2.langdetect,
            cfg.o.vtc2.word_count,
        ),
        "whisperx": (
            str(WHISPERX / "langdetect_fallback" / "langdetect_tags.py"),
            str(WHISPERX / "langdetect_fallback" / "count_words.py"),
            cfg.o.whisperx.tagged,
            cfg.o.whisperx.langdetect,
            cfg.o.whisperx.word_count,
        ),
    }

    for pipe_name, (tagger, counter, input_dir, lang_dir, count_dir) in langdetect_configs.items():
        if pipe_name not in pipelines:
            continue
        if not check(input_dir, f"{pipe_name} transcripts for langdetect"):
            continue

        print(f"\n  [4a] Language tag correction — {pipe_name}...")
        ok &= run([
            sys.executable, tagger,
            "-i", input_dir,
            "-o", lang_dir,
        ], f"langdetect_tags_{pipe_name}", dry)

        print(f"\n  [4b] Word count analysis — {pipe_name}...")
        ok &= run([
            sys.executable, counter,
            "-i", lang_dir,
            "-o", count_dir,
        ], f"count_words_{pipe_name}", dry)

    return ok


# ─────────────────────────────────────────────────────────────────
# STEP 5 — Excel Report Generation
# ─────────────────────────────────────────────────────────────────

def step5_excel(cfg, dry):
    step_header(5, "Excel Report Generation")
    ok = True

    out = cfg.output

    # 5a: WER comparison Excel (per-clip detail)
    print("\n  [5a] Generating WER comparison Excel (per-clip)...")
    ok &= run([
        sys.executable,
        str(PROJ / "wer_per_recording_excel.py"),
    ], "pipeline_comparision_eval_final", dry)

    # 5b: WER comparison Excel (per-recording summary)
    print("\n  [5b] Generating WER comparison Excel (per-recording)...")
    ok &= run([
        sys.executable,
        str(PROJ / "wer_per_clip_excel.py"),
    ], "36_clip_excel", dry)

    # 5c: Word count comparison Excel
    print("\n  [5c] Generating language distribution comparison Excel...")
    ok &= run([
        sys.executable,
        str(PROJ / "word_count_comparison_excel.py"),
    ], "word_count_comparison_excel", dry)

    return ok


# ─────────────────────────────────────────────────────────────────
# CONFIG OBJECT — all output paths derived from output root
# ─────────────────────────────────────────────────────────────────

class PipelinePaths:
    """Holds all intermediate and final output paths."""
    def __init__(self, root):
        r = Path(root)

        class Human:
            benchmark  = str(r / "human_eval" / "five_mins_benchmark")
            combined   = str(r / "human_eval" / "combined")
            word_count = str(r / "human_eval" / "word_count")

        class VTC1:
            rttm_txt          = str(r / "vtc1" / "rttm_txt")
            tagged            = str(r / "vtc1" / "tagged_transcripts")
            speaker_separated = str(r / "vtc1" / "speaker_separated")
            eval_report       = str(r / "vtc1" / "eval_report")
            langdetect        = str(r / "vtc1" / "langdetect")
            word_count        = str(r / "vtc1" / "word_count")

        class VTC2:
            rttm_raw          = str(r / "vtc2" / "rttm_raw")     # VTC2 model raw output
            rttm_txt          = str(r / "vtc2" / "rttm_txt")
            tagged            = str(r / "vtc2" / "tagged_transcripts")
            speaker_separated = str(r / "vtc2" / "speaker_separated")
            eval_report       = str(r / "vtc2" / "eval_report")
            langdetect        = str(r / "vtc2" / "langdetect")
            word_count        = str(r / "vtc2" / "word_count")

        class WhisperX:
            chunks            = str(r / "whisperx_only" / "chunks")
            raw_transcripts   = str(r / "whisperx_only" / "raw_transcripts")
            merged            = str(r / "whisperx_only" / "merged")
            tagged            = str(r / "whisperx_only" / "tagged_transcripts")
            eval_report       = str(r / "whisperx_only" / "eval_report")
            langdetect        = str(r / "whisperx_only" / "langdetect")
            word_count        = str(r / "whisperx_only" / "word_count")

        self.human   = Human()
        self.vtc1    = VTC1()
        self.vtc2    = VTC2()
        self.whisperx = WhisperX()


class Config:
    def __init__(self, args):
        self.audio      = str(args.audio)
        self.annotated  = str(args.annotated)
        self.output     = str(args.output)
        self.vtc1_model = str(args.vtc1_model)
        self.vtc2_model = str(args.vtc2_model)
        self.language   = args.language
        self.device     = args.device
        self.o          = PipelinePaths(args.output)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end ASR pipeline: audio + annotations → Excel reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--audio",      default=str(PROJ / "audio"),
                        help="Input audio directory (default: audio/)")
    parser.add_argument("--annotated",  default=str(PROJ / "annotated-text"),
                        help="Human annotated transcripts directory (default: annotated-text/)")
    parser.add_argument("--output",     default=str(PROJ / "outputs"),
                        help="Root output directory (default: outputs/)")
    parser.add_argument("--vtc1-model",
                        default=os.environ.get("ASR_VTC1_MODEL"), required="ASR_VTC1_MODEL" not in os.environ,
                        help="VTC1 model directory containing apply.sh")
    parser.add_argument("--vtc2-model",
                        default=os.environ.get("ASR_VTC2_MODEL"), required="ASR_VTC2_MODEL" not in os.environ,
                        help="VTC2 model directory containing scripts/infer.py")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "gpu", "mps"],
                        help="Device for VTC2 inference (default: cpu)")
    parser.add_argument("--pipelines",  nargs="+", default=["vtc1", "vtc2", "whisperx"],
                        choices=["vtc1", "vtc2", "whisperx"],
                        help="Which pipelines to run (default: all)")
    parser.add_argument("--steps",      nargs="+", type=int, default=[0,1,2,3,4,5],
                        help="Which steps to run (default: all 0-5)")
    parser.add_argument("--language",   default="Spanish",
                        choices=["Spanish", "English"],
                        help="Transcript language for evaluation (default: Spanish)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    cfg = Config(args)
    dry = args.dry_run
    steps = set(args.steps)
    pipelines = set(args.pipelines)

    print("=" * 70)
    print("  ASR PIPELINE ORCHESTRATOR")
    print("=" * 70)
    print(f"  Audio dir     : {cfg.audio}")
    print(f"  Annotated dir : {cfg.annotated}")
    print(f"  Output root   : {cfg.output}")
    print(f"  Pipelines     : {', '.join(sorted(pipelines))}")
    print(f"  Steps         : {sorted(steps)}")
    print(f"  Language      : {cfg.language}")
    print(f"  VTC1 model    : {cfg.vtc1_model}")
    print(f"  VTC2 model    : {cfg.vtc2_model}")
    print(f"  Device        : {cfg.device}")
    if dry:
        print(f"  MODE          : DRY RUN (no commands will execute)")
    print("=" * 70)

    # Validate inputs
    if not Path(cfg.audio).exists():
        print(f"\nERROR: Audio directory not found: {cfg.audio}")
        sys.exit(1)
    if not Path(cfg.annotated).exists():
        print(f"\nERROR: Annotated text directory not found: {cfg.annotated}")
        sys.exit(1)

    # Create output root
    mkdir(cfg.output)

    results = {}

    if 0 in steps:
        results[0] = step0_human_eval(cfg, dry)

    if 1 in steps:
        step_header("pre-1", "Audio Format Check — Normalise to 16kHz Mono")
        ensure_16k_mono(cfg.audio, dry)

        if "vtc1" in pipelines:
            results["1_vtc1"] = step1_vtc1(cfg, dry)
        if "vtc2" in pipelines:
            results["1_vtc2"] = step1_vtc2(cfg, dry)
        if "whisperx" in pipelines:
            results["1_whisperx"] = step1_whisperx(cfg, dry)

    if 2 in steps:
        results[2] = step2_evaluate(cfg, pipelines, dry)

    if 3 in steps:
        results[3] = step3_aggregate(cfg, pipelines, dry)

    if 4 in steps:
        results[4] = step4_langdetect(cfg, pipelines, dry)

    if 5 in steps:
        results[5] = step5_excel(cfg, dry)

    # Final summary
    print(f"\n{'='*70}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*70}")
    all_ok = all(v for v in results.values())
    if all_ok:
        print("   All steps completed successfully")
        print(f"\n  Output files:")
        print(f"    {cfg.output}/")
        print(f"    wer_per_recording.xlsx")
        print(f"    wer_per_clip.xlsx")
        print(f"    word_count_comparison.xlsx")
    else:
        failed = [str(k) for k, v in results.items() if not v]
        print(f"  WARNING: Some steps had issues: {', '.join(failed)}")
        print(f"  Check output above for details.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
