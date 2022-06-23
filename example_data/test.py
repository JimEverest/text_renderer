import inspect
import os
from pathlib import Path
import imgaug.augmenters as iaa
# from text_renderer.corpus.customize_HKID_CTC_corpus import CustomizeHKID_CTC_Corpus, CustomizeHKID_CTC_CorpusCfg
                                                           
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
FONT_DIR = DATA_DIR / "font_hk_digits"
FONT_LIST_DIR = DATA_DIR / "font_list"
TEXT_DIR = DATA_DIR / "text"

font_cfg = dict(
    font_dir=FONT_DIR,
    # font_list_file=FONT_LIST_DIR / "font_list.txt",
    font_size=(30, 33),
)

perspective_transform = NormPerspectiveTransformCfg(20, 20, 1.5)

# default:
def get_char_corpus():
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[TEXT_DIR / "chn_text.txt", TEXT_DIR / "eng_text.txt"],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "chn.txt",
            length=(5, 10),
            # char_spacing=(-0.3, 1.3),
            char_spacing=(0, 0.4),
            **font_cfg
        ),
    )



# WordCorpus---> continous words.
def eng_word_data():
    return WordCorpus(
        WordCorpusCfg(
            text_paths=[TEXT_DIR / "en2.txt"],
            separator=" ",
            num_word = (2,5),
            filter_by_chars=False,
            text_color_cfg =FixedTextColorCfg(),
            # chars_file=CHAR_DIR / "eng.txt",
            **font_cfg
        )
    )


# 466479_enum_en
# 60000_top

# 80768_americanbanker
# 4853_reversedictionary

# AOF
# SAO_EN






def base_cfg(name: str, corpus, corpus_effects=None, layout_effects=None, layout=None, gray=True, render_effects=None, num=50):
    return GeneratorCfg(
        num_image=num,
        save_dir=OUT_DIR / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            perspective_transform=perspective_transform,
            gray=gray,
            layout_effects=layout_effects,
            layout=layout,
            corpus=corpus,
            corpus_effects=corpus_effects,
            render_effects=render_effects,
            height = 32
        ),
    )




#========================== Corpus ==========================
#0. customised HKID CTC digits corpus:
def get_HK_CTC_corpus():
    return CustomizeHKID_CTC_Corpus(
            CustomizeHKID_CTC_CorpusCfg(
            chars_file=TEXT_DIR / "10Digits.txt", 
            length=(5, 10),
#         filter_by_chars=True,
#         chars_file=TEXT_DIR / "en_dict.txt",
            char_spacing=(-0.05, 0.2),
            text_color_cfg =FixedTextColorCfg(),
            **font_cfg),
    )

    


# 1. Rand: ----> chars
def get_en_char_corpus():
    return RandCorpus(
            RandCorpusCfg(
            chars_file=TEXT_DIR / "96_char_en_dict.txt", 
            length=(5, 15),
#         filter_by_chars=True,
#         chars_file=TEXT_DIR / "en_dict.txt",
            char_spacing=(-0.05, 0.2),
            text_color_cfg =FixedTextColorCfg(),
            **font_cfg),
    )

# 1.1 Rand: ----> chars  [CN]
def get_cn_char_corpus():
    return RandCorpus(
            RandCorpusCfg(
            chars_file=TEXT_DIR / "10Digits_4Symbols.txt", #"96_char_en_dict.txt", #"hkid_all_v1_999.txt", #"187_confusion_chars.txt", #"96rare_300FirstName.txt",
            length=(4, 12),
#         filter_by_chars=True,
            char_spacing=(-0.05, 0.2),
            text_color_cfg =FixedTextColorCfg(),
            **font_cfg),
    )


# 2.randomly get words from dict Line.
def bk_enum_corpus():
    return EnumCorpus(
            EnumCorpusCfg(
                text_paths=[TEXT_DIR / "AOF.txt",TEXT_DIR / "SAO_EN.txt"],
                num_pick = (1,5),
                join_str =" ",
                char_spacing=(-0.05, 0.2),
                text_color_cfg =FixedTextColorCfg(),
                filter_by_chars=True,
                chars_file=TEXT_DIR / "96_char_en_dict.txt", 
                **font_cfg
            )
    )

# 3.[WIDE] randomly get words from dict Line.
def bk_enum_wide_corpus():
    return EnumCorpus(
            EnumCorpusCfg(
                text_paths=[TEXT_DIR / "80768_americanbanker.txt",TEXT_DIR / "4853_reversedictionary.txt"],
                num_pick = (1,5),
                join_str =" ",
                char_spacing=(-0.05, 0.2),
                text_color_cfg =FixedTextColorCfg(),
                filter_by_chars=True,
                chars_file=TEXT_DIR / "96_char_en_dict.txt", 
                **font_cfg
            )
    )

#========================== Config ==========================
# 1. Rand: ----> chars
def random_chars_500000():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        num=500000,
        corpus=get_en_char_corpus(),
        # corpus=enum_data(),
        layout=ExtraTextLineLayout(bottom_prob=1.0),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.0, 0.21], h_ratio=[0.0, 0.21], center=False),
                Line(p=0.2, thickness=(1, 3), line_pos_p=(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0)),
                # Emboss(p=0.9,alpha=(0.9, 1.0), strength=(1.5, 1.6)),
                # DropoutRand(p=1, dropout_p=(0.3, 0.5)),
                DropoutHorizontal(p=0.5, num_line=2, thickness=1),
                MotionBlur(p=0.1, k=(3, 3), angle=(0, 360), direction=(-1.0, 1.0)),
                
                Snow(p=0,level=1),
                CoarseDropout(p=0.7,noise=0.03, size_percent=0.5),
                # JpegCompression(level=2)
                # ImgAugEffect(aug=iaa.imgcorruptlike.JpegCompression(severity=1))
                # SnowFlakes(),
            ]
        ), 
        render_effects=Effects(
            [
                SaltAndPepper(p=0.3,noise=0.003),
                SnowFlakes(p=0.2),
                JpegCompression(level=2)
            ]
        )
    )

    random_chars_30w_NumChars(),
    random_chars_30w_NumChars(),
   
# 1.1 Rand: ----> chars 【CN】
def hk_ctc_digits_10w():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        num=100000,
        corpus=get_HK_CTC_corpus(),
        # corpus=enum_data(),
        layout=ExtraTextLineLayout(bottom_prob=1.0),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.0, 0.21], h_ratio=[0.0, 0.21], center=False),
                Line(p=0.2, thickness=(1, 3), line_pos_p=(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0)),
                # Emboss(p=0.9,alpha=(0.9, 1.0), strength=(1.5, 1.6)),
                # DropoutRand(p=1, dropout_p=(0.3, 0.5)),
                DropoutHorizontal(p=0.5, num_line=2, thickness=1),
                MotionBlur(p=0.1, k=(3, 3), angle=(0, 360), direction=(-1.0, 1.0)),
                
                Snow(p=0,level=1),
                CoarseDropout(p=0.7,noise=0.03, size_percent=0.5),
                # JpegCompression(level=2)
                # ImgAugEffect(aug=iaa.imgcorruptlike.JpegCompression(severity=1))
                # SnowFlakes(),
            ]
        ), 
        render_effects=Effects(
            [
                SaltAndPepper(p=0.3,noise=0.003),
                SnowFlakes(p=0.2),
                JpegCompression(level=2)
            ]
        )
    )




# 1.1 Rand: ----> chars 【CN】
def random_chars_20w_NumChars():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        num=200000,
        corpus=get_cn_char_corpus(),
        # corpus=enum_data(),
        layout=ExtraTextLineLayout(bottom_prob=1.0),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.0, 0.21], h_ratio=[0.0, 0.21], center=False),
                Line(p=0.2, thickness=(1, 3), line_pos_p=(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0)),
                # Emboss(p=0.9,alpha=(0.9, 1.0), strength=(1.5, 1.6)),
                # DropoutRand(p=1, dropout_p=(0.3, 0.5)),
                DropoutHorizontal(p=0.5, num_line=2, thickness=1),
                MotionBlur(p=0.1, k=(3, 3), angle=(0, 360), direction=(-1.0, 1.0)),
                
                Snow(p=0,level=1),
                CoarseDropout(p=0.7,noise=0.03, size_percent=0.5),
                # JpegCompression(level=2)
                # ImgAugEffect(aug=iaa.imgcorruptlike.JpegCompression(severity=1))
                # SnowFlakes(),
            ]
        ), 
        render_effects=Effects(
            [
                SaltAndPepper(p=0.3,noise=0.003),
                SnowFlakes(p=0.2),
                JpegCompression(level=2)
            ]
        )
    )


# 2.randomly get words from dict Line.

def bank_chars_aof_1000000():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        num=1000000,
        corpus=bk_enum_corpus(),
        # corpus=enum_data(),
        layout=ExtraTextLineLayout(bottom_prob=1.0),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.0, 0.21], h_ratio=[0.0, 0.21], center=False),
                Line(p=0.2, thickness=(1, 3), line_pos_p=(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0)),
                # Emboss(p=0.9,alpha=(0.9, 1.0), strength=(1.5, 1.6)),
                # DropoutRand(p=1, dropout_p=(0.3, 0.5)),
                DropoutHorizontal(p=0.5, num_line=2, thickness=1),
                MotionBlur(p=0.1, k=(3, 3), angle=(0, 360), direction=(-1.0, 1.0)),
                
                Snow(p=0,level=1),
                CoarseDropout(p=0.7,noise=0.03, size_percent=0.5),
                # JpegCompression(level=2)
                # ImgAugEffect(aug=iaa.imgcorruptlike.JpegCompression(severity=1))
                # SnowFlakes(),
            ]
        ), 
        render_effects=Effects(
            [
                SaltAndPepper(p=0.3,noise=0.003),
                SnowFlakes(p=0.2),
                JpegCompression(level=2)
            ]
        )
    )

#3.
def bank_chars_wide_1000000():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        num=1000000,
        corpus=bk_enum_wide_corpus(),
        # corpus=enum_data(),
        layout=ExtraTextLineLayout(bottom_prob=1.0),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.0, 0.21], h_ratio=[0.0, 0.21], center=False),
                Line(p=0.2, thickness=(1, 3), line_pos_p=(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0)),
                # Emboss(p=0.9,alpha=(0.9, 1.0), strength=(1.5, 1.6)),
                # DropoutRand(p=1, dropout_p=(0.3, 0.5)),
                DropoutHorizontal(p=0.3, num_line=2, thickness=1),
                DropoutVertical(p=0.2, num_line=3, thickness=1),
                MotionBlur(p=0.1, k=(3, 3), angle=(0, 360), direction=(-1.0, 1.0)),
                
                Snow(p=0,level=1),
                CoarseDropout(p=0.7,noise=0.03, size_percent=0.5),
                # JpegCompression(level=2)
                # ImgAugEffect(aug=iaa.imgcorruptlike.JpegCompression(severity=1))
                # SnowFlakes(),
            ]
        ), 
        render_effects=Effects(
            [
                SaltAndPepper(p=0.3,noise=0.003),
                SnowFlakes(p=0.2),
                JpegCompression(level=2)
            ]
        )
    )


#4. for anir
def chars_20000():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        num=20000,
        corpus=bk_enum_wide_corpus(),
        # corpus=enum_data(),
        layout=ExtraTextLineLayout(bottom_prob=1.0),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.0, 0.21], h_ratio=[0.0, 0.21], center=False),
                Line(p=0.2, thickness=(1, 3), line_pos_p=(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0)),
                # Emboss(p=0.01,alpha=(0.9, 1.0), strength=(1.5, 1.6)),
                DropoutHorizontal(p=0.3, num_line=2, thickness=1),
                # DropoutVertical(p=0.2, num_line=3, thickness=1),
                # MotionBlur(p=0.05, k=(3, 3), angle=(0, 360), direction=(-1.0, 1.0)),
                # Snow(p=0,level=1),
                # CoarseDropout(p=0.7,noise=0.03, size_percent=0.5),

                # DropoutRand(p=1, dropout_p=(0.3, 0.5)),
                # JpegCompression(level=2)
                # ImgAugEffect(aug=iaa.imgcorruptlike.JpegCompression(severity=1))
                SnowFlakes(),
            ]
        ), 
        render_effects=Effects(
            [
                SaltAndPepper(p=0.4,noise=0.003),
                SnowFlakes(p=0.2),
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

    # random_chars_500000(),
    # bank_chars_aof_1000000(),
    # bank_chars_wide_1000000(),
    random_chars_20w_NumChars(),
    # hk_ctc_digits_10w(),
    # chars_20000()
]
# fmt: on


# python3 main.py --config example_data/text.py --dataset img --num_processes 2 --log_period 10






#todo:

#doing:

#done:
#1. emboss
#2. snow
# 1. visualize
# 2. lbl converter











def imgaug_emboss_example2():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
            ]
        ), 
        render_effects=Effects(
            [
                JpegCompression(level=2),
                SnowFlakes()
            ]
        )
    )

    # line_poses = [
    #     "top",
    #     "bottom",
    #     "left",
    #     "right",
    #     "top_left",
    #     "top_right",
    #     "bottom_left",
    #     "bottom_right",
    #     "horizontal_middle",
    #     "vertical_middle",
    # ]

    # line_pos_p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    # line_pos_p (:obj:`tuple`) : Each value corresponds a line position. Must sum to 1.
    #             top, bottom, left, right, top_left, top_right, bottom_left, bottom_right, horizontal_middle, vertical_middle
