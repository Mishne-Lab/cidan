from LSSC.functions.pickle_funcs import *
def test_pickle_funcs():
    test_dir = "test_pickle"
    pickle_set_dir(test_dir)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)
    pickle_clear(trial_num=0)
    assert not pickle_exist("test", trial_num=0)
    obj = "pickle save"
    pickle_save(obj, "test",trial_num=0)
    assert len([f for f in os.listdir("{0}/0/".format(test_dir))]) == 1
    assert pickle_load("test", trial_num=0)==obj
    assert pickle_exist("test", trial_num=0)
    pickle_clear(trial_num=0)
    assert not pickle_exist("test", trial_num=0)
    assert len([f for f in os.listdir("{0}/0/".format(test_dir))]) == 0
    os.rmdir("{0}/0/".format(test_dir))
    os.rmdir(test_dir)

