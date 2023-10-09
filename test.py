import argparse




if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--name","-n", default="sunny", type=str) #conda create --prefix env python=3.0 or we use conda create -p venv python=3.0
    args.add_argument("--age", "-a", default=25, type=float)
    parse_args=args.parse_args()
    
    print(parse_args.name,parse_args.age)