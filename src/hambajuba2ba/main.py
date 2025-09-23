"""Main entry point for hambajuba2ba"""
from imagegen import get_noise_img

def main():
   img = get_noise_img(256)
   img.save('test.png')


if __name__ == "__main__":
    main()
