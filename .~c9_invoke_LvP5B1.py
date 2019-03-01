import argparse
import os.path 

def run_nst(content_path, style_path):
    '''  ''' 
    pass 
def main(a, b):
def an_interesting_function(content_path, style_path):
    directory = os.path.dirname(os.path.abspath(content_path))
    #directory_list = os.listdir(directory)
    new_dirname = os.path.join(directory, 'new_imgs')
    os.mkdir(new_dirname)
    for file in directory: 
        run_nst(content_path, style_path) 
        new_image_filename = os.path.join(new_directory, filename + '.png')
        new_image.save(new_image_filename)

    

def main(a, b):
    print(a+b)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args()
    main(int(args.a), int(args.b))