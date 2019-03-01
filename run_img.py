import utils
import argparse


def main(content_path, style_path, num_iterations=1000):
    model = utils.Model()
    images = utils.ContentAndStyleImage(content_path, style_path)
    best_img, best_loss = model.run_style_transfer(
        images, num_iterations=num_iterations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_path", "-c", type=str)
    parser.add_argument("--style_path", "-s", type=str)
    parser.add_argument("--num_iter", "-i", type=int, required=False)

    args = parser.parse_args()
    main(args.content_path, args.style_path, args.num_iter)
