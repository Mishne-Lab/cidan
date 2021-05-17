import eigen_display
import fire
import os

def generate_eigen_vectors(folder,shape=(235,235)):
    runs = list(os.listdir(folder))
    for run in runs:
        if os.path.isfile(os.path.join(folder, run,"embedding_norm_images/embedding_norm_image.png")):
            try:
                eigen_display.display_eigen(os.path.join(folder,run,"eigen_vectors/"), os.path.join(folder,run,"eigen_norm.png"), shape=shape)
                print("success")
            except:
                print("Fail")


if __name__ == '__main__':
    fire.Fire(generate_eigen_vectors)