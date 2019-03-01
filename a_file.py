import argparse
import os.path 

def run_nst(content_path, style_path):
    ''' docstring :) ''' 
    pass 
    
def modify(directory, style_path):
    ''' docstring :) ''' 
    #obtains content image and all other images in its directory 
    # directory = os.path.dirname(os.path.abspath(content_path))
    
    # new directory to store modified images 
    new_dir = os.path.join(directory, 'modified')
    os.mkdir(new_dir)
    
    # for every file, modify and place in new directory 
    for file in directory: 
        new_img = run_nst(file, style_path)
        (head, tail) = os.path.split(file)
        new_img_filename = os.path.join(new_dir, f"{tail}_modified") # f"{tail}_modified"
        new_img.save(new_img_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("style_path")
    args = parser.parse_args()
    modify(args.directory, args.style_path)