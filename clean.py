import argparse
import os
from glob import glob



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="example_data/test.py")
    parser.add_argument("--output", default="example_data/output/")
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--log_period", type=float, default=10)
    return parser.parse_args()


if __name__=="__main__":

    args = parse_args()
    # args.output=args.output
    #1. CLEAN OUTPUT
    # os.system("rm -r "+ args.output+"*")

    #2. Generate
    os.system("python /home/jim/AI/text_renderer/main.py --config "+args.config+" --dataset img --num_processes 6 --log_period 10")

    #3. Convert LBL
    os.system("python lbl_conv.py --look /home/jim/AI/text_renderer/example_data/output/")
    
    #4. Vis Random Result
    pp = glob(args.output+"*/", recursive = True)
    print(pp)

    for pth in pp:
        print("pth---->", pth) # example_data/output/chars_2000/
        os.system("python vis.py --look "+ pth + "images")    #--look example_data/output/chars_2000/images




