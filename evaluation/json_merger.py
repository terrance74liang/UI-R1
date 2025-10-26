import json
import argparse

def merger(output_name:str, kwargs):
    annotations = []
    for file_name in kwargs:
        with open(file_name, 'r') as f:
            annotations += json.load(f)
    
    with open(output_name,'w') as f:
        json.dump(annotations,f)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o','--output')
    parser.add_argument('-i','--inputs', nargs='*')
    parsed_args = parser.parse_args()
    print("OUTPUT: " + parsed_args.output)
    print("INPUT: " + str(parsed_args.inputs))
    merger(parsed_args.output,parsed_args.inputs)
