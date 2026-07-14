"""Command-line options for ArtGate inference and evaluation."""

from __future__ import annotations

import argparse


DEFAULT_TESTSETS = (
    "progan,stylegan,biggan,cyclegan,stargan,gaugan,stylegan2,"
    "whichfaceisreal,ADM,Glide,Midjourney,stable_diffusion_v_1_4,"
    "stable_diffusion_v_1_5,VQDM,wukong,DALLE2,sd_xl"
)


class TestOptions:
    """Parse the options used by ``ArtGate_eval.py``."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._add_arguments()

    def _add_arguments(self):
        parser = self.parser

        # Inputs and outputs
        parser.add_argument(
            "--model_path",
            default="./weights/model_artgate_progan.pth",
            help="complete ArtGate checkpoint",
        )
        parser.add_argument(
            "--image_path",
            default=None,
            help="single image path; skips dataset evaluation when provided",
        )
        parser.add_argument(
            "--dataset_root",
            default="./datasets",
            help="directory containing one folder per test set",
        )
        parser.add_argument("--testsets", default=DEFAULT_TESTSETS)
        parser.add_argument("--results_dir", default="./results/ArtGate")

        # Runtime
        parser.add_argument("--device", default="cuda", help="cuda or cuda:N")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--max_test_image", type=int, default=None)
        parser.add_argument("--fc_class2", action="store_true")

        # Image preprocessing
        parser.add_argument("--loadSize", type=int, default=256)
        parser.add_argument("--CropSize", type=int, default=224)
        parser.add_argument("--no_crop", action="store_true")
        parser.add_argument("--no_resize", action="store_true")
        parser.add_argument("--rz_interp", default="bilinear")

        # Optional robustness perturbations
        parser.add_argument(
            "--noise_type",
            default=None,
            help="jpeg, jpg, blur, resize, webp, or another supported perturbation",
        )
        parser.add_argument("--jpeg_quality_min", type=int, default=75)
        parser.add_argument("--jpeg_quality_max", type=int, default=95)
        parser.add_argument("--jpg_method", default="pil")
        parser.add_argument("--jpg_qual", default="95")

    def parse(self, print_options=True):
        opt = self.parser.parse_args()
        opt.isTrain = False
        opt.isVal = False

        opt.rz_interp = self._split_nonempty(opt.rz_interp)
        opt.jpg_method = self._split_nonempty(opt.jpg_method)
        opt.jpg_qual = self._parse_quality_values(opt.jpg_qual)
        opt.testsets = self._split_nonempty(opt.testsets)

        if not opt.testsets and opt.image_path is None:
            self.parser.error("--testsets must contain at least one dataset name")
        if opt.batch_size < 1:
            self.parser.error("--batch_size must be at least 1")
        if opt.max_test_image is not None and opt.max_test_image < 1:
            self.parser.error("--max_test_image must be at least 1")
        if opt.jpeg_quality_min > opt.jpeg_quality_max:
            self.parser.error("--jpeg_quality_min cannot exceed --jpeg_quality_max")

        if print_options:
            self.print_options(opt)
        return opt

    def print_options(self, opt):
        lines = ["----------------- Options ---------------"]
        for name, value in sorted(vars(opt).items()):
            default = self.parser.get_default(name)
            suffix = f"\t[default: {default}]" if value != default else ""
            lines.append(f"{name:>25}: {str(value):<30}{suffix}")
        lines.append("----------------- End -------------------")
        print("\n".join(lines))

    @staticmethod
    def _split_nonempty(value):
        return [item.strip() for item in value.split(",") if item.strip()]

    def _parse_quality_values(self, value):
        try:
            qualities = [int(item) for item in self._split_nonempty(value)]
        except ValueError as error:
            self.parser.error(f"--jpg_qual must contain integers: {error}")

        if not qualities:
            self.parser.error("--jpg_qual cannot be empty")
        if len(qualities) == 2:
            start, end = qualities
            if start > end:
                self.parser.error("--jpg_qual range must be ascending")
            qualities = list(range(start, end + 1))
        elif len(qualities) > 2:
            self.parser.error("--jpg_qual accepts one value or an inclusive range")
        if any(not 1 <= quality <= 100 for quality in qualities):
            self.parser.error("JPEG quality values must be between 1 and 100")
        return qualities
