import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-a",action="store_true")
args = parser.parse_args()
print(type(args.a))
print(args.a)