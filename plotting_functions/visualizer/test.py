from html import HTML

from visualizer import Visualizer

visualizer = Visualizer()
# create website
web_dir = "path_of_webpage"
webpage = HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    visuals = model.get_current_visuals()  # a tuple of label and numpy image
    img_path = "path_of_image"
    print('%04d: process image... %s' % (i, img_path))
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
