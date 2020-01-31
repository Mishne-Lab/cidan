from LSSC.Stack import Stack
from LSSC.Parameters import Parameters
from LSSC.functions.pickle_funcs import *
def test_stack():
    test_dir = "test_stack"
    pickle_set_dir(test_dir)
    pickle_clear("test")
    data_stack = Stack(
        "/data2/Sam/pythonTestEnviroment/input_images/8_6_14_d10_001.tif", "test",
        Parameters(num_threads=10))
    clusters = data_stack.clusters(5, save_images = False)
    assert len(clusters) == 5
    assert type(clusters) == list
    image_path = test_dir+"/test.png"
    embeding_image = data_stack.embeding_image(image_path)
    assert os.path.isfile(image_path)
    os.unlink(image_path)
    pickle_clear("test")
    for filename in os.listdir(os.path.join(test_dir,"test")):
            os.unlink(filename)
    os.rmdir(os.path.join(test_dir,"test"))

    os.rmdir(test_dir)


