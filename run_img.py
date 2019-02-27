import utils
import argparse


def main(content_path, style_path):
    model = utils.Model()
    images = utils.ContentAndStyleImage(content_path, style_path)
    best_img, best_loss = model.run_style_transfer(images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("content_path")
    parser.add_argument("style_path")
    args = parser.parse_args()
    main(args.content_path, args.style_path)
