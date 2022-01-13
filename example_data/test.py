import inspect
import os
from pathlib import Path
import imgaug.augmenters as iaa

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    FixedTextColorCfg,
)
from text_renderer.layout.same_line import SameLineLayout
from text_renderer.layout.extra_text_line import ExtraTextLineLayout


CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__))) #/home/jim/AI/text_renderer/example_data/
HOME_DIR = CURRENT_DIR.parent
OUT_DIR = CURRENT_DIR / "output"


DATA_DIR = CURRENT_DIR
BG_DIR = DATA_DIR / "bg"
CHAR_DIR = DATA_DIR / "char"
FONT_DIR = DATA_DIR / "font"
FONT_LIST_DIR = DATA_DIR / "font_list"
TEXT_DIR = DATA_DIR / "text"

font_cfg = dict(
    font_dir=FONT_DIR,
    font_list_file=FONT_LIST_DIR / "font_list.txt",
    font_size=(30, 31),
)

perspective_transform = NormPerspectiveTransformCfg(20, 20, 1.5)


def get_char_corpus():
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[TEXT_DIR / "chn_text.txt", TEXT_DIR / "eng_text.txt"],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "chn.txt",
            length=(5, 10),
            char_spacing=(-0.3, 1.3),
            **font_cfg
        ),
    )


def base_cfg(name: str, corpus, corpus_effects=None, layout_effects=None, layout=None, gray=True, render_effects=None):
    return GeneratorCfg(
        num_image=50,
        save_dir=OUT_DIR / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            perspective_transform=perspective_transform,
            gray=gray,
            layout_effects=layout_effects,
            layout=layout,
            corpus=corpus,
            corpus_effects=corpus_effects,
            render_effects=render_effects
        ),
    )




def imgaug_emboss_example():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                # Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                # Emboss(p=0.9,alpha=(0.9, 1.0), strength=(1.5, 1.6)),
                # CoarseDropout(p=1.0,noise=0.99, size_percent=1.0),
                # JpegCompression(level=2)
                # ImgAugEffect(aug=iaa.imgcorruptlike.JpegCompression(severity=1))
                # SnowFlakes(),
            ]
        ), 
        render_effects=Effects(
            [
                JpegCompression(level=2)
            ]
        )
    )

# fmt: off
# The configuration file must have a configs variable
configs = [
    # chn_data(),
    # enum_data(),
    # rand_data(),
    # eng_word_data(),
    # same_line_data(),
    # extra_text_line_data(),
    imgaug_emboss_example()
]
# fmt: on


# python3 main.py --config example_data/text.py --dataset img --num_processes 2 --log_period 10






#todo:
# 1. visualize
# 2. lbl converter

#doing
#1. emboss
#2. snow

#done:
#
#
#

